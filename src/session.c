#include "model.h"
#include "session.h"
#include "backend_session.h"
#include "transformer_internal.h"
#include "turboquant.h"
#include "sh_log.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int checked_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t tmp;
    if (checked_mul_size(a, b, &tmp) != 0) return -1;
    return checked_mul_size(tmp, c, out);
}

static int checked_mul4_size(size_t a, size_t b, size_t c, size_t d, size_t *out) {
    size_t tmp;
    if (checked_mul3_size(a, b, c, &tmp) != 0) return -1;
    return checked_mul_size(tmp, d, out);
}

BnSession *bn_session_create(const BnModel *model, BnAllocator *alloc) {
    if (!model) return NULL;
    const BnConfig *c = &model->config;

    // Allocate session struct
    BnSession *s;
    if (alloc) {
        s = (BnSession *)bn_malloc(alloc, sizeof(BnSession));
    } else {
        s = (BnSession *)malloc(sizeof(BnSession));
    }
    if (!s) return NULL;
    memset(s, 0, sizeof(BnSession));
    s->backend = bn_backend_session_create();
    if (!s->backend) {
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Create session arena
    size_t arena_size = bn_model_session_arena_size(c, &model->weights);
    if (arena_size > SIZE_MAX / 2) {
        bn_backend_session_free(s->backend);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->arena = sh_arena_create(arena_size);
    if (!s->arena) {
        SH_LOG_ERROR("Failed to allocate session arena");
        bn_backend_session_free(s->backend);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    // Allocate session buffers
    if (bn_model_alloc_session_buffers(c, &model->weights, s->arena,
                                        &s->state, &s->moe_state) != 0) {
        SH_LOG_ERROR("Failed to allocate session buffers");
        sh_arena_free(s->arena);
        bn_backend_session_free(s->backend);
        if (alloc) bn_free(alloc, s, sizeof(BnSession));
        else free(s);
        return NULL;
    }

    s->pos = 0;
    s->gpu_kv_direct_valid = 0;
    s->gpu_ssm_direct_valid = 0;
    return s;
}

void bn_session_free(BnSession *s, BnAllocator *alloc) {
    if (!s) return;
    bn_backend_session_free(s->backend);
    if (s->state.key_cache && s->state.key_cache_alloc_bytes > 0)
        munmap(s->state.key_cache, s->state.key_cache_alloc_bytes);
    if (s->state.value_cache && s->state.value_cache_alloc_bytes > 0)
        munmap(s->state.value_cache, s->state.value_cache_alloc_bytes);
    sh_arena_free(s->arena);
    if (alloc) {
        bn_free(alloc, s, sizeof(BnSession));
    } else {
        free(s);
    }
}

static int bn_session_ssm_sizes(const BnModel *model,
                                size_t *out_state_floats,
                                size_t *out_conv_floats) {
    if (!model) return -1;
    const BnConfig *c = &model->config;
    if (c->full_attn_interval <= 0 || c->ssm_time_step_rank <= 0) {
        *out_state_floats = 0;
        *out_conv_floats = 0;
        return -1;
    }
    int n_attn = c->n_layers / c->full_attn_interval;
    int n_ssm = c->n_layers - n_attn;
    if (n_ssm <= 0) {
        *out_state_floats = 0;
        *out_conv_floats = 0;
        return -1;
    }
    int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
    size_t state_per_layer;
    if (checked_mul3_size((size_t)c->ssm_time_step_rank,
                          (size_t)c->ssm_state_size,
                          (size_t)head_v_dim, &state_per_layer) != 0)
        return -1;
    size_t state_total;
    if (checked_mul_size((size_t)n_ssm, state_per_layer, &state_total) != 0)
        return -1;
    int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
    size_t conv_total;
    if (checked_mul3_size((size_t)n_ssm,
                          (size_t)(c->ssm_conv_kernel - 1),
                          (size_t)conv_dim, &conv_total) != 0)
        return -1;
    *out_state_floats = state_total;
    *out_conv_floats = conv_total;
    return 0;
}

size_t bn_session_recurrent_state_bytes(const BnModel *model) {
    size_t state_floats = 0, conv_floats = 0;
    if (bn_session_ssm_sizes(model, &state_floats, &conv_floats) != 0)
        return 0;
    if (state_floats > SIZE_MAX - conv_floats) return 0;
    size_t total_floats = state_floats + conv_floats;
    size_t total_bytes;
    if (checked_mul_size(total_floats, sizeof(float), &total_bytes) != 0)
        return 0;
    return total_bytes;
}

int bn_session_get_recurrent_state(const BnSession *s, const BnModel *model,
                                   void *out, size_t out_bytes) {
    if (!s || !model || !out) return -1;
    size_t state_floats = 0, conv_floats = 0;
    if (bn_session_ssm_sizes(model, &state_floats, &conv_floats) != 0)
        return -1;
    size_t expected = (state_floats + conv_floats) * sizeof(float);
    if (out_bytes != expected) return -1;
    if (!s->state.ssm_state || !s->state.ssm_conv_state) return -1;

    if (bn_model_gpu((BnModel *)model)) {
        bn_transformer_gpu_download_ssm_state((BnModel *)model,
                                              (BnSession *)s);
    }

    uint8_t *dst = (uint8_t *)out;
    memcpy(dst, s->state.ssm_state, state_floats * sizeof(float));
    memcpy(dst + state_floats * sizeof(float),
           s->state.ssm_conv_state, conv_floats * sizeof(float));
    return 0;
}

int bn_session_set_recurrent_state(BnSession *s, const BnModel *model,
                                   const void *in, size_t in_bytes) {
    if (!s || !model || !in) return -1;
    size_t state_floats = 0, conv_floats = 0;
    if (bn_session_ssm_sizes(model, &state_floats, &conv_floats) != 0)
        return -1;
    size_t expected = (state_floats + conv_floats) * sizeof(float);
    if (in_bytes != expected) return -1;
    if (!s->state.ssm_state || !s->state.ssm_conv_state) return -1;
    const uint8_t *src = (const uint8_t *)in;
    memcpy(s->state.ssm_state, src, state_floats * sizeof(float));
    memcpy(s->state.ssm_conv_state,
           src + state_floats * sizeof(float),
           conv_floats * sizeof(float));

    if (bn_model_gpu((BnModel *)model)) {
        if (bn_transformer_gpu_upload_ssm_state((BnModel *)model, s) != 0)
            return -1;
    }
    return 0;
}

void bn_session_kv_truncate(BnSession *s, int new_pos) {
    if (!s || new_pos < 0) return;
    if (new_pos > s->pos) return;
    s->pos = new_pos;
    s->gpu_kv_direct_valid = 0;
    s->gpu_ssm_direct_valid = 0;
}

void bn_session_reset(BnSession *s, const BnModel *model) {
    if (!s || !model) return;
    const BnConfig *c = &model->config;
    BnRunState *rs = &s->state;

    // KV cache
    int n_attn = (c->full_attn_interval > 0)
        ? c->n_layers / c->full_attn_interval : c->n_layers;
    size_t kv_size = 0;
    if (n_attn < 0 ||
        checked_mul3_size((size_t)n_attn, (size_t)c->seq_len,
                          (size_t)c->kv_dim, &kv_size) != 0)
        return;
    size_t kv_elem = c->kv_f16 ? sizeof(uint16_t) : sizeof(float);
    size_t kv_bytes = 0;
    if (checked_mul_size(kv_size, kv_elem, &kv_bytes) != 0) return;
    if (rs->key_cache_alloc_bytes > 0)
        (void)madvise(rs->key_cache, rs->key_cache_alloc_bytes, MADV_DONTNEED);
    else
        memset(rs->key_cache, 0, kv_bytes);
    if (rs->value_cache_alloc_bytes > 0)
        (void)madvise(rs->value_cache, rs->value_cache_alloc_bytes, MADV_DONTNEED);
    else
        memset(rs->value_cache, 0, kv_bytes);

    // TQ compressed KV cache
    if (rs->key_cache_tq && rs->value_cache_tq && c->kv_tq_bits > 0 && bn_model_tq_state(model)) {
        int kb = bn_tq_key_bytes(bn_model_tq_state(model));
        int vb = bn_tq_value_bytes(bn_model_tq_state(model));
        size_t tq_key_total = 0, tq_val_total = 0;
        if (checked_mul4_size((size_t)n_attn, (size_t)c->seq_len,
                              (size_t)c->n_kv_heads, (size_t)kb, &tq_key_total) != 0 ||
            checked_mul4_size((size_t)n_attn, (size_t)c->seq_len,
                              (size_t)c->n_kv_heads, (size_t)vb, &tq_val_total) != 0)
            return;
        memset(rs->key_cache_tq, 0, tq_key_total);
        memset(rs->value_cache_tq, 0, tq_val_total);
    }

    // SSM state
    if (rs->ssm_state && c->ssm_time_step_rank > 0) {
        int n_ssm = c->n_layers - n_attn;
        int head_v_dim = c->ssm_inner_size / c->ssm_time_step_rank;
        size_t state_total = 0;
        if (n_ssm < 0 ||
            checked_mul4_size((size_t)n_ssm, (size_t)c->ssm_time_step_rank,
                              (size_t)c->ssm_state_size, (size_t)head_v_dim,
                              &state_total) != 0 ||
            checked_mul_size(state_total, sizeof(float), &state_total) != 0)
            return;
        memset(rs->ssm_state, 0, state_total);
    }
    if (rs->ssm_conv_state) {
        int n_ssm = c->n_layers - n_attn;
        int conv_dim = c->ssm_group_count * c->ssm_state_size * 2 + c->ssm_inner_size;
        size_t conv_total = 0;
        if (n_ssm < 0 ||
            checked_mul4_size((size_t)n_ssm, (size_t)(c->ssm_conv_kernel - 1),
                              (size_t)conv_dim, sizeof(float), &conv_total) != 0)
            return;
        memset(rs->ssm_conv_state, 0, conv_total);
    }

    s->pos = 0;
    s->gpu_kv_direct_valid = 0;
    s->gpu_ssm_direct_valid = 0;
}
