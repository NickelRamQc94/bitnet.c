/*
 * gpu_metal.m — Native Metal compute backend for BnGPUBackend
 *
 * Implements the BnGPUBackend vtable using Apple Metal.
 * Unified memory (storageModeShared) — no staging buffers.
 * setBytes for uniforms — no ring buffer.
 * Runtime shader compilation from .metal source files.
 * precise transcendentals for SSM IEEE compliance.
 */

#ifdef BN_ENABLE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu_metal.h"
#include "gpu_backend.h"
#include "gpu_shader.h"
#include "model.h"
#include "quant.h"
#include "gguf.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* Max tensor type enum value we index into (I2_S = 36, plus margin) */
#define BN_METAL_MAX_TYPES 40

/* ── Internal context ──────────────────────────────────────────────── */

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pipelines[BN_METAL_MAX_TYPES];  /* matvec per quant type */
    id<MTLComputePipelineState> fwd_pipelines[BN_GPU_SHADER_COUNT]; /* forward-pass shaders */
    id<MTLComputePipelineState> q8_quant_pipeline;
    id<MTLComputePipelineState> q8k_quant_pipeline;
    id<MTLComputePipelineState> q4_q8_matvec_pipeline;
    id<MTLComputePipelineState> q4_prepared_q8_matvec_pipeline;
    id<MTLComputePipelineState> q4_prepared_q8_split_pipeline;
    id<MTLComputePipelineState> q4_prepared_q8_gateup_pipeline;
    id<MTLComputePipelineState> q4_q8_split_pipeline;
    id<MTLComputePipelineState> q4_q8_gateup_pipeline;
    id<MTLComputePipelineState> cpu_order_rmsnorm_pipeline;
    id<MTLComputePipelineState> q6_q8k_matvec_pipeline;
    id<MTLComputePipelineState> ssm_prefill_rmsnorm_pipeline;
    id<MTLComputePipelineState> ssm_prefill_conv_silu_pipeline;
    id<MTLComputePipelineState> ssm_prefill_l2norm_pipeline;
    id<MTLComputePipelineState> ssm_prefill_alpha_beta_pipeline;
    id<MTLComputePipelineState> ssm_prefill_delta_pipeline;
    id<MTLComputePipelineState> ssm_prefill_delta_precise_pipeline;
    id<MTLComputePipelineState> ssm_prefill_delta_fused_ab_pipeline;
    id<MTLComputePipelineState> ssm_prefill_delta_fused_full_pipeline;
    id<MTLComputePipelineState> ssm_prefill_gate_pipeline;
    id<MTLComputePipelineState> ssm_prefill_silu_gate_stacked_pipeline;
    id<MTLComputePipelineState> q4k_mul_mm_pipeline;
    id<MTLComputePipelineState> q5k_mul_mm_pipeline;
    id<MTLComputePipelineState> q6k_mul_mm_pipeline;
    id<MTLComputePipelineState> q8_0_mul_mm_pipeline;
    id<MTLComputePipelineState> q4_0_mul_mm_pipeline;
    id<MTLComputePipelineState> q4_1_mul_mm_pipeline;
    id<MTLComputePipelineState> q3k_mul_mm_pipeline;
    id<MTLComputePipelineState> q2k_mul_mm_pipeline;
    id<MTLComputePipelineState> f16_mul_mm_pipeline;
    id<MTLComputePipelineState> bf16_mul_mm_pipeline;
    id<MTLComputePipelineState> iq4_nl_mul_mm_pipeline;
    id<MTLComputePipelineState> iq4_xs_mul_mm_pipeline;
    id<MTLComputePipelineState> iq3_xxs_mul_mm_pipeline;
    id<MTLComputePipelineState> iq3_s_mul_mm_pipeline;
    id<MTLComputePipelineState> iq2_s_mul_mm_pipeline;
    id<MTLComputePipelineState> iq2_xxs_mul_mm_pipeline;
    int  q4k_mul_mm_spec_K[16];
    bool q4k_mul_mm_spec_bc_out[16];
    id<MTLComputePipelineState> q4k_mul_mm_spec_pipeline[16];
    int  q5k_mul_mm_spec_K[16];
    bool q5k_mul_mm_spec_bc_out[16];
    id<MTLComputePipelineState> q5k_mul_mm_spec_pipeline[16];
    int  q6k_mul_mm_spec_K[16];
    bool q6k_mul_mm_spec_bc_out[16];
    id<MTLComputePipelineState> q6k_mul_mm_spec_pipeline[16];
    id<MTLComputePipelineState> prefill_attn_pipeline;
    id<MTLComputePipelineState> prefill_attn_pipeline_6144;
    id<MTLComputePipelineState> prefill_kv_prep_pipeline;
    int ssm_prefill_enabled;
    int q4_q8_enabled;
    int q8_barriers_enabled;
    int cpu_order_rmsnorm_enabled;

    /* GPU-resident activation buffers (storageModeShared) */
    id<MTLBuffer> act_bufs[BN_GPU_BUF_COUNT];
    size_t        act_sizes[BN_GPU_BUF_COUNT];

    /* Persistent scratch buffers for standalone matvec */
    id<MTLBuffer> x_buf;
    size_t        x_buf_size;
    id<MTLBuffer> out_buf;
    size_t        out_buf_size;
    id<MTLBuffer> q8_buf;
    size_t        q8_buf_size;
    id<MTLBuffer> q8_scales_buf;
    size_t        q8_scales_buf_size;
    id<MTLBuffer> q8_bsums_buf;
    size_t        q8_bsums_buf_size;
    id<MTLBuffer> ssm_prefill_buf;
    size_t        ssm_prefill_buf_size;
    id<MTLBuffer>                  batch_act_buf[2];
    size_t                         batch_act_buf_size[2];
    int                            batch_chain_index;
    int                            batch_active;
    id<MTLCommandBuffer>           batch_cmd;
    id<MTLComputeCommandEncoder>   batch_enc;
    id<MTLFence>                   batch_fence;
    id<MTLBuffer>                  gpu_key_cache;
    id<MTLBuffer>                  gpu_value_cache;
    size_t                         gpu_kv_cache_bytes;
    int                            gpu_kv_n_layers;
    int                            gpu_kv_seq_len;
    int                            gpu_kv_dim;
    /* Shader directory path */
    char shader_dir[256];

    /* Profiling */
    int gpu_frame;
    int gpu_profile;
    int q8_quant_dispatches;
    int q8k_quant_dispatches;
    int q8_matvec_dispatches;
    int q8_split_dispatches;
    int q8_gateup_dispatches;

    /* Slab allocator for MoE weight suballocation */
    id<MTLBuffer> slab_buf;
    size_t        slab_size;
    struct { size_t offset, size; } *slab_free;
    int           slab_free_count;
    int           slab_free_cap;

    /* Zero-copy mmap range (Phase 5) */
    const void   *mmap_base;
    size_t        mmap_size;
} BnMetalCtx;

typedef struct {
    int shader;
    uint32_t rows;
    uint32_t cols;
    uint32_t aux;
    int count;
    double gpu_ms;
    double wall_ms;
} BnMetalProfileShape;

static const char *metal_shader_profile_name(int shader)
{
    static const char *names[] = {
        "matvec","rmsnorm","rope","gqa_scores","softmax","gqa_combine",
        "silu_gate","relu2_gate","resid_add","copy","bias_add","resid_rmsnorm",
        "weighted_add","ssm_conv","ssm_l2norm","ssm_ab","ssm_delta","ssm_gate",
        "per_head_norm","deinterleave_q","sigmoid_gate","flash_attn",
        "matvec_split","rope_qk","fused_gateup","ssm_ab_split","q4k_split"
    };
    if (shader >= 0 && shader < (int)(sizeof(names) / sizeof(names[0])))
        return names[shader];
    return "?";
}

static void metal_profile_add_shape(BnMetalProfileShape *shapes,
                                    int *n_shapes,
                                    int max_shapes,
                                    int shader,
                                    uint32_t rows,
                                    uint32_t cols,
                                    uint32_t aux)
{
    for (int i = 0; i < *n_shapes; i++) {
        if (shapes[i].shader == shader && shapes[i].rows == rows &&
            shapes[i].cols == cols && shapes[i].aux == aux) {
            shapes[i].count++;
            return;
        }
    }
    if (*n_shapes >= max_shapes) return;
    shapes[*n_shapes].shader = shader;
    shapes[*n_shapes].rows = rows;
    shapes[*n_shapes].cols = cols;
    shapes[*n_shapes].aux = aux;
    shapes[*n_shapes].count = 1;
    shapes[*n_shapes].gpu_ms = 0.0;
    shapes[*n_shapes].wall_ms = 0.0;
    (*n_shapes)++;
}

static void metal_profile_add_shape_time(BnMetalProfileShape *shapes,
                                         int *n_shapes,
                                         int max_shapes,
                                         int shader,
                                         uint32_t rows,
                                         uint32_t cols,
                                         uint32_t aux,
                                         double gpu_ms,
                                         double wall_ms)
{
    for (int i = 0; i < *n_shapes; i++) {
        if (shapes[i].shader == shader && shapes[i].rows == rows &&
            shapes[i].cols == cols && shapes[i].aux == aux) {
            shapes[i].count++;
            shapes[i].gpu_ms += gpu_ms;
            shapes[i].wall_ms += wall_ms;
            return;
        }
    }
    if (*n_shapes >= max_shapes) return;
    shapes[*n_shapes].shader = shader;
    shapes[*n_shapes].rows = rows;
    shapes[*n_shapes].cols = cols;
    shapes[*n_shapes].aux = aux;
    shapes[*n_shapes].count = 1;
    shapes[*n_shapes].gpu_ms = gpu_ms;
    shapes[*n_shapes].wall_ms = wall_ms;
    (*n_shapes)++;
}

/* ── GPU buffer handle ─────────────────────────────────────────────── */

typedef struct {
    id<MTLBuffer> buf;
    size_t        size;
    size_t        offset;       /* byte offset into slab (0 for standalone) */
    int           type;
    int           rows;
    int           cols;
    uint32_t      bias_offset;  /* u32 offset for fused bias, 0 = none */
    int           is_slab;
    int           q4_repacked;
    int           q4_prepared;
} BnMetalBuf;

static int metal_q4_prepared_upload_enabled(void)
{
    const char *from_layer = getenv("BN_GPU_Q4_Q8_FROM_LAYER");
    return getenv("BN_METAL_Q4_PREPARED") &&
           getenv("BN_GPU_Q4_Q8") &&
           (!from_layer || atoi(from_layer) <= 0) &&
           !getenv("BN_GPU_Q4_Q8_ATTN_ONLY") &&
           !getenv("BN_GPU_Q4_Q8_FFN_ONLY");
}

/* ── Shader type name mapping (same as wgpu) ──────────────────────── */

static const char *shader_name_for_type(int type)
{
    switch (type) {
        case BN_GGUF_TENSOR_F32:     return "f32";
        case BN_GGUF_TENSOR_F16:     return "f16";
        case BN_GGUF_TENSOR_I2_S:    return "i2s";
        case BN_GGUF_TENSOR_TQ1_0:   return "tq1";
        case BN_GGUF_TENSOR_TQ2_0:   return "tq2";
        case BN_GGUF_TENSOR_Q4_0:    return "q4";
        case BN_GGUF_TENSOR_Q4_1:    return "q4_1";
        case BN_GGUF_TENSOR_Q8_0:    return "q8";
        case BN_GGUF_TENSOR_BF16:    return "bf16";
        case BN_GGUF_TENSOR_Q2_K:    return "q2k";
        case BN_GGUF_TENSOR_Q3_K:    return "q3k";
        case BN_GGUF_TENSOR_Q4_K:    return "q4k";
        case BN_GGUF_TENSOR_Q5_K:    return "q5k";
        case BN_GGUF_TENSOR_Q6_K:    return "q6k";
        case BN_GGUF_TENSOR_Q8_K:    return "q8k";
        case BN_GGUF_TENSOR_IQ4_NL:  return "iq4nl";
        case BN_GGUF_TENSOR_IQ4_XS:  return "iq4xs";
        case BN_GGUF_TENSOR_IQ3_XXS: return "iq3xxs";
        case BN_GGUF_TENSOR_IQ3_S:   return "iq3s";
        case BN_GGUF_TENSOR_IQ2_XXS: return "iq2xxs";
        case BN_GGUF_TENSOR_IQ2_XS:  return "iq2xs";
        case BN_GGUF_TENSOR_IQ2_S:   return "iq2s";
        default:                      return NULL;
    }
}

static const int supported_types[] = {
    BN_GGUF_TENSOR_I2_S, BN_GGUF_TENSOR_TQ1_0, BN_GGUF_TENSOR_TQ2_0,
    BN_GGUF_TENSOR_Q4_0, BN_GGUF_TENSOR_Q4_1, BN_GGUF_TENSOR_Q8_0,
    BN_GGUF_TENSOR_F16, BN_GGUF_TENSOR_BF16, BN_GGUF_TENSOR_Q2_K, BN_GGUF_TENSOR_Q3_K,
    BN_GGUF_TENSOR_Q4_K, BN_GGUF_TENSOR_Q5_K, BN_GGUF_TENSOR_Q6_K,
    BN_GGUF_TENSOR_Q8_K, BN_GGUF_TENSOR_IQ4_NL, BN_GGUF_TENSOR_IQ4_XS,
    BN_GGUF_TENSOR_IQ3_XXS, BN_GGUF_TENSOR_IQ3_S, BN_GGUF_TENSOR_IQ2_XXS,
    BN_GGUF_TENSOR_IQ2_XS, BN_GGUF_TENSOR_IQ2_S,
};
#define N_SUPPORTED_TYPES ((int)(sizeof(supported_types) / sizeof(supported_types[0])))

/* ── Shader compilation ────────────────────────────────────────────── */

static id<MTLComputePipelineState> compile_shader(BnMetalCtx *ctx,
                                                   const char *dir,
                                                   const char *filename,
                                                   const char *fn_name)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);

    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:nsPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
    if (!source) return nil;

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:source
                                                   options:opts
                                                     error:&err];
    if (!lib) {
        fprintf(stderr, "[bn:gpu:metal] shader compile error (%s): %s\n",
                filename, [[err localizedDescription] UTF8String]);
        return nil;
    }

    NSString *fnName = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [lib newFunctionWithName:fnName];
    if (!fn) {
        fprintf(stderr, "[bn:gpu:metal] function '%s' not found in %s\n",
                fn_name, filename);
        return nil;
    }

    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
    if (!pso) {
        fprintf(stderr, "[bn:gpu:metal] pipeline error (%s): %s\n",
                filename, [[err localizedDescription] UTF8String]);
    }
    return pso;
}

static id<MTLComputePipelineState>
compile_shader_with_fc(BnMetalCtx *ctx, const char *dir,
                       const char *filename, const char *fn_name,
                       MTLFunctionConstantValues *fc)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, filename);
    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSError *err = nil;
    NSString *source = [NSString stringWithContentsOfFile:nsPath
                                                 encoding:NSUTF8StringEncoding
                                                    error:&err];
    if (!source) return nil;

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:source
                                                   options:opts
                                                     error:&err];
    if (!lib) return nil;

    NSString *fnName = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [lib newFunctionWithName:fnName
                                   constantValues:fc
                                            error:&err];
    if (!fn) {
        fprintf(stderr, "[bn:gpu:metal] FC function '%s' not found in %s: %s\n",
                fn_name, filename,
                err ? [[err localizedDescription] UTF8String] : "(no err)");
        return nil;
    }

    id<MTLComputePipelineState> pso =
        [ctx->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[bn:gpu:metal] FC pipeline error (%s): %s\n",
                filename, [[err localizedDescription] UTF8String]);
    }
    return pso;
}

static int compile_matvec_pipeline(BnMetalCtx *ctx, int type, const char *dir)
{
    const char *name = shader_name_for_type(type);
    if (!name) return -1;
    if (type < 0 || type >= BN_METAL_MAX_TYPES) return -1;

    char filename[64], fn_name[64];
    snprintf(filename, sizeof(filename), "%s_matvec.metal", name);
    snprintf(fn_name, sizeof(fn_name), "%s_matvec", name);

    id<MTLComputePipelineState> pso = compile_shader(ctx, dir, filename, fn_name);
    if (!pso) return -1;

    ctx->pipelines[type] = pso;
    return 0;
}

static id<MTLBuffer> metal_new_weight_buffer(BnMetalCtx *ctx,
                                             const void *data,
                                             size_t size)
{
    if (!ctx || !data || size == 0) return nil;
    if (getenv("BN_METAL_SHARED_WEIGHTS")) {
        return [ctx->device newBufferWithBytes:data
                                        length:size
                                       options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> dst = [ctx->device newBufferWithLength:size
                                                  options:MTLResourceStorageModePrivate];
    id<MTLBuffer> staging = [ctx->device newBufferWithBytes:data
                                                     length:size
                                                    options:MTLResourceStorageModeShared];
    if (!dst || !staging) return nil;

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if ([cmd status] == MTLCommandBufferStatusError) return nil;
    return dst;
}

/* ── Slab allocator ────────────────────────────────────────────────── */

static int slab_init(BnMetalCtx *ctx, size_t size)
{
    if (ctx->slab_buf || size == 0) return -1;
    ctx->slab_buf = [ctx->device newBufferWithLength:size
                                             options:MTLResourceStorageModeShared];
    if (!ctx->slab_buf) return -1;
    ctx->slab_size = size;
    ctx->slab_free_cap = 256;
    ctx->slab_free = calloc((size_t)ctx->slab_free_cap,
                            sizeof(ctx->slab_free[0]));
    if (!ctx->slab_free) return -1;
    ctx->slab_free[0].offset = 0;
    ctx->slab_free[0].size = size;
    ctx->slab_free_count = 1;
    return 0;
}

static size_t slab_alloc(BnMetalCtx *ctx, size_t size)
{
    size = (size + 255) & ~(size_t)255;  /* 256-byte align */
    for (int i = 0; i < ctx->slab_free_count; i++) {
        if (ctx->slab_free[i].size >= size) {
            size_t offset = ctx->slab_free[i].offset;
            ctx->slab_free[i].offset += size;
            ctx->slab_free[i].size -= size;
            if (ctx->slab_free[i].size == 0) {
                ctx->slab_free[i] = ctx->slab_free[--ctx->slab_free_count];
            }
            return offset;
        }
    }
    return (size_t)-1;
}

static void slab_free_range(BnMetalCtx *ctx, size_t offset, size_t size)
{
    if (ctx->slab_free_count >= ctx->slab_free_cap) {
        ctx->slab_free_cap *= 2;
        ctx->slab_free = realloc(ctx->slab_free,
                          (size_t)ctx->slab_free_cap * sizeof(ctx->slab_free[0]));
    }
    ctx->slab_free[ctx->slab_free_count].offset = offset;
    ctx->slab_free[ctx->slab_free_count].size = size;
    ctx->slab_free_count++;
}

/* ── Vtable: buffer_create ─────────────────────────────────────────── */

static BnMetalBuf *metal_repack_q4_0_for_gpu(BnMetalCtx *ctx,
                                             const void *data,
                                             size_t size,
                                             int rows,
                                             int cols,
                                             const float *bias,
                                             int bias_len,
                                             int allow_prepared)
{
    (void)size;
    if (!ctx || !data || rows <= 0 || cols <= 0 || (cols % 32) != 0)
        return NULL;

    int blocks_per_row = cols / 32;
    int n_blocks = rows * blocks_per_row;
    if (allow_prepared && metal_q4_prepared_upload_enabled() && (rows % 4) == 0) {
        int n_groups = rows / 4;
        size_t n_group_blocks = (size_t)n_groups * (size_t)blocks_per_row;
        size_t scale_bytes = n_group_blocks * 4 * sizeof(uint16_t);
        size_t qs_offset = (scale_bytes + 3) & ~(size_t)3;
        size_t qs_bytes = n_group_blocks * 64;
        size_t bias_bytes = (bias && bias_len > 0)
            ? (size_t)bias_len * sizeof(float) : 0;
        size_t bias_offset_bytes = qs_offset + qs_bytes;
        size_t prepared_size = (bias_offset_bytes + bias_bytes + 3) & ~(size_t)3;
        uint8_t *prepared_data = (uint8_t *)calloc(1, prepared_size);
        if (!prepared_data) return NULL;

        uint16_t *scales = (uint16_t *)prepared_data;
        uint8_t *qs_out = prepared_data + qs_offset;
        const BnBlockQ4_0 *blocks = (const BnBlockQ4_0 *)data;
        for (int g = 0; g < n_groups; g++) {
            for (int b = 0; b < blocks_per_row; b++) {
                size_t gb = (size_t)g * (size_t)blocks_per_row + (size_t)b;
                for (int r = 0; r < 4; r++) {
                    size_t src = (size_t)(g * 4 + r) *
                                 (size_t)blocks_per_row + (size_t)b;
                    scales[gb * 4 + (size_t)r] = blocks[src].d;
                }
                uint8_t *dst = qs_out + gb * 64;
                for (int ng = 0; ng < 4; ng++) {
                    for (int r = 0; r < 4; r++) {
                        size_t src = (size_t)(g * 4 + r) *
                                     (size_t)blocks_per_row + (size_t)b;
                        const uint8_t *qs = blocks[src].qs + ng * 4;
                        uint8_t *dp = dst + ng * 16 + r * 4;
                        for (int j = 0; j < 4; j++)
                            dp[j] = qs[j] ^ 0x88;
                    }
                }
            }
        }
        uint32_t bias_offset = 0;
        if (bias && bias_len > 0) {
            bias_offset = (uint32_t)(bias_offset_bytes / sizeof(uint32_t));
            memcpy(prepared_data + bias_offset_bytes, bias,
                   (size_t)bias_len * sizeof(float));
        }

        BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
        if (!buf) {
            free(prepared_data);
            return NULL;
        }
        buf->buf = metal_new_weight_buffer(ctx, prepared_data, prepared_size);
        free(prepared_data);
        if (!buf->buf) {
            free(buf);
            return NULL;
        }
        buf->size = prepared_size;
        buf->offset = 0;
        buf->type = BN_GGUF_TENSOR_Q4_0;
        buf->rows = rows;
        buf->cols = cols;
        buf->bias_offset = bias_offset;
        buf->q4_prepared = 1;
        buf->q4_repacked = 1;
        return buf;
    }

    size_t base_size = (size_t)n_blocks * sizeof(float) +
                       (size_t)n_blocks * 4 * sizeof(uint32_t);
    size_t bias_bytes = (bias && bias_len > 0) ?
                        (size_t)bias_len * sizeof(float) : 0;
    size_t repacked_size = (base_size + bias_bytes + 3) & ~(size_t)3;

    uint8_t *repacked = (uint8_t *)calloc(1, repacked_size);
    if (!repacked) return NULL;

    float *scales = (float *)repacked;
    uint8_t *nibbles = repacked + (size_t)n_blocks * sizeof(float);
    const uint8_t *src = (const uint8_t *)data;

    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + (size_t)b * 18;
        uint16_t d_bits = (uint16_t)(block[0] | (block[1] << 8));
        scales[b] = bn_fp16_to_fp32(d_bits);

        uint8_t *dst_nib = nibbles + (size_t)b * 16;
        const uint8_t *qs = block + 2;
        dst_nib[0]  = (qs[0] & 0x0F) | ((qs[1] & 0x0F) << 4);
        dst_nib[1]  = (qs[2] & 0x0F) | ((qs[3] & 0x0F) << 4);
        dst_nib[2]  = (qs[4] & 0x0F) | ((qs[5] & 0x0F) << 4);
        dst_nib[3]  = (qs[6] & 0x0F) | ((qs[7] & 0x0F) << 4);
        dst_nib[4]  = (qs[8] & 0x0F) | ((qs[9] & 0x0F) << 4);
        dst_nib[5]  = (qs[10] & 0x0F) | ((qs[11] & 0x0F) << 4);
        dst_nib[6]  = (qs[12] & 0x0F) | ((qs[13] & 0x0F) << 4);
        dst_nib[7]  = (qs[14] & 0x0F) | ((qs[15] & 0x0F) << 4);
        dst_nib[8]  = (qs[0] >> 4) | ((qs[1] >> 4) << 4);
        dst_nib[9]  = (qs[2] >> 4) | ((qs[3] >> 4) << 4);
        dst_nib[10] = (qs[4] >> 4) | ((qs[5] >> 4) << 4);
        dst_nib[11] = (qs[6] >> 4) | ((qs[7] >> 4) << 4);
        dst_nib[12] = (qs[8] >> 4) | ((qs[9] >> 4) << 4);
        dst_nib[13] = (qs[10] >> 4) | ((qs[11] >> 4) << 4);
        dst_nib[14] = (qs[12] >> 4) | ((qs[13] >> 4) << 4);
        dst_nib[15] = (qs[14] >> 4) | ((qs[15] >> 4) << 4);
    }

    uint32_t bias_offset = 0;
    if (bias && bias_len > 0) {
        bias_offset = (uint32_t)(base_size / sizeof(uint32_t));
        memcpy(repacked + base_size, bias, (size_t)bias_len * sizeof(float));
    }

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) {
        free(repacked);
        return NULL;
    }
    buf->buf = metal_new_weight_buffer(ctx, repacked, repacked_size);
    free(repacked);
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = repacked_size;
    buf->offset = 0;
    buf->type = BN_GGUF_TENSOR_Q4_0;
    buf->rows = rows;
    buf->cols = cols;
    buf->bias_offset = bias_offset;
    buf->q4_repacked = 1;
    return buf;
}

static void *metal_buffer_create(void *vctx, const void *data, size_t size,
                                  int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0) return NULL;

    if (type == BN_GGUF_TENSOR_Q4_0)
        return metal_repack_q4_0_for_gpu(ctx, data, size, rows, cols, NULL, 0, 1);

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) return NULL;

    /* Try slab allocation first */
    if (ctx->slab_buf) {
        size_t aligned = (size + 255) & ~(size_t)255;
        size_t offset = slab_alloc(ctx, aligned);
        if (offset != (size_t)-1) {
            memcpy((uint8_t *)[ctx->slab_buf contents] + offset, data, size);
            buf->buf = ctx->slab_buf;
            buf->size = size;
            buf->offset = offset;
            buf->type = type;
            buf->rows = rows;
            buf->cols = cols;
            buf->is_slab = 1;
            return buf;
        }
    }

    /* Zero-copy: if data is within mmap range, wrap it without copying.
     * Requires page-aligned pointer. The mmap'd file stays alive for the
     * lifetime of the model, so the buffer is valid. */
    if (ctx->mmap_base && ctx->mmap_size > 0) {
        const uint8_t *base = (const uint8_t *)ctx->mmap_base;
        const uint8_t *ptr = (const uint8_t *)data;
        if (ptr >= base && ptr + size <= base + ctx->mmap_size) {
            /* Page-align: extend range to page boundaries */
            size_t page = (size_t)getpagesize();
            uintptr_t aligned_start = (uintptr_t)ptr & ~(page - 1);
            size_t prefix = (uintptr_t)ptr - aligned_start;
            size_t aligned_size = (prefix + size + page - 1) & ~(page - 1);
            buf->buf = [ctx->device newBufferWithBytesNoCopy:(void *)aligned_start
                                                      length:aligned_size
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
            if (buf->buf) {
                buf->offset = prefix;
                buf->size = size;
                buf->type = type;
                buf->rows = rows;
                buf->cols = cols;
                buf->is_slab = 0;
                return buf;
            }
            /* Fall through to copy path if NoCopy fails */
        }
    }

    buf->buf = metal_new_weight_buffer(ctx, data, size);
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    buf->size = size;
    buf->offset = 0;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    buf->is_slab = 0;
    return buf;
}

static void *metal_buffer_create_biased(void *vctx, const void *data, size_t size,
                                         int type, int rows, int cols,
                                         const void *bias, size_t bias_size)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || size == 0 || !bias || bias_size == 0) return NULL;

    if (type == BN_GGUF_TENSOR_Q4_0) {
        int bias_len = (int)(bias_size / sizeof(float));
        return metal_repack_q4_0_for_gpu(ctx, data, size, rows, cols,
                                         (const float *)bias, bias_len, 1);
    }

    /* Other types: combine weight data + bias into one buffer */
    size_t total = size + bias_size;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data, size);
    memcpy(combined + size, bias, bias_size);

    BnMetalBuf *buf = (BnMetalBuf *)metal_buffer_create(vctx, combined, total,
                                                          type, rows, cols);
    free(combined);
    if (!buf) return NULL;

    buf->bias_offset = (uint32_t)(size / sizeof(uint32_t));
    return buf;
}

static void *metal_buffer_create_stacked2(void *vctx,
                                          const void *data0, size_t size0,
                                          const void *data1, size_t size1,
                                          int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data0 || !data1 || size0 == 0 || size1 == 0) return NULL;

    size_t total = size0 + size1;
    if (type == BN_GGUF_TENSOR_Q4_0) {
        uint8_t *combined = (uint8_t *)malloc(total);
        if (!combined) return NULL;
        memcpy(combined, data0, size0);
        memcpy(combined + size0, data1, size1);
        BnMetalBuf *buf = metal_repack_q4_0_for_gpu(
            ctx, combined, total, rows, cols, NULL, 0, 0);
        free(combined);
        return buf;
    }

    BnMetalBuf *buf = (BnMetalBuf *)calloc(1, sizeof(BnMetalBuf));
    if (!buf) return NULL;

    if (ctx->slab_buf) {
        size_t aligned = (total + 255) & ~(size_t)255;
        size_t offset = slab_alloc(ctx, aligned);
        if (offset != (size_t)-1) {
            uint8_t *dst = (uint8_t *)[ctx->slab_buf contents] + offset;
            memcpy(dst, data0, size0);
            memcpy(dst + size0, data1, size1);
            buf->buf = ctx->slab_buf;
            buf->size = total;
            buf->offset = offset;
            buf->type = type;
            buf->rows = rows;
            buf->cols = cols;
            buf->is_slab = 1;
            return buf;
        }
    }

    buf->buf = [ctx->device newBufferWithLength:total
                                        options:MTLResourceStorageModeShared];
    if (!buf->buf) {
        free(buf);
        return NULL;
    }
    uint8_t *dst = (uint8_t *)[buf->buf contents];
    memcpy(dst, data0, size0);
    memcpy(dst + size0, data1, size1);
    buf->size = total;
    buf->offset = 0;
    buf->type = type;
    buf->rows = rows;
    buf->cols = cols;
    return buf;
}

static void *metal_buffer_create_stacked3(void *vctx,
                                          const void *data0, size_t size0,
                                          const void *data1, size_t size1,
                                          const void *data2, size_t size2,
                                          int type, int rows, int cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data0 || !data1 || !data2 ||
        size0 == 0 || size1 == 0 || size2 == 0)
        return NULL;
    if (type == BN_GGUF_TENSOR_Q4_0 && metal_q4_prepared_upload_enabled())
        return NULL;

    size_t total = size0 + size1 + size2;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    memcpy(combined + size0 + size1, data2, size2);

    BnMetalBuf *buf = NULL;
    if (type == BN_GGUF_TENSOR_Q4_0) {
        buf = metal_repack_q4_0_for_gpu(
            ctx, combined, total, rows, cols, NULL, 0, 0);
    } else {
        buf = (BnMetalBuf *)metal_buffer_create(
            vctx, combined, total, type, rows, cols);
    }
    free(combined);
    return buf;
}

static void *metal_buffer_create_stacked3_biased(void *vctx,
                                                 const void *data0,
                                                 size_t size0,
                                                 const void *data1,
                                                 size_t size1,
                                                 const void *data2,
                                                 size_t size2,
                                                 int type,
                                                 int rows,
                                                 int cols,
                                                 const void *bias,
                                                 size_t bias_size)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data0 || !data1 || !data2 || !bias ||
        size0 == 0 || size1 == 0 || size2 == 0 || bias_size == 0)
        return NULL;
    if (type == BN_GGUF_TENSOR_Q4_0 && metal_q4_prepared_upload_enabled())
        return NULL;

    size_t total = size0 + size1 + size2;
    uint8_t *combined = (uint8_t *)malloc(total);
    if (!combined) return NULL;
    memcpy(combined, data0, size0);
    memcpy(combined + size0, data1, size1);
    memcpy(combined + size0 + size1, data2, size2);

    BnMetalBuf *buf = NULL;
    if (type == BN_GGUF_TENSOR_Q4_0) {
        int bias_len = (int)(bias_size / sizeof(float));
        buf = metal_repack_q4_0_for_gpu(
            ctx, combined, total, rows, cols, (const float *)bias,
            bias_len, 0);
    } else {
        size_t combined_biased_size = total + bias_size;
        uint8_t *combined_biased = (uint8_t *)malloc(combined_biased_size);
        if (combined_biased) {
            memcpy(combined_biased, combined, total);
            memcpy(combined_biased + total, bias, bias_size);
            buf = (BnMetalBuf *)metal_buffer_create(
                vctx, combined_biased, combined_biased_size, type, rows, cols);
            if (buf)
                buf->bias_offset = (uint32_t)(total / sizeof(uint32_t));
            free(combined_biased);
        }
    }
    free(combined);
    return buf;
}

static void metal_buffer_destroy(void *vctx, void *buffer)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *buf = (BnMetalBuf *)buffer;
    if (!buf) return;

    if (buf->is_slab && ctx) {
        slab_free_range(ctx, buf->offset, (buf->size + 255) & ~(size_t)255);
    }
    /* Standalone buffers: ARC releases when buf->buf goes out of scope */
    /* (Under ARC, setting to nil or letting it deallocate handles release) */
    free(buf);
}

/* ── Vtable: init_activations ──────────────────────────────────────── */

static void metal_free_activations(void *vctx);  /* forward decl */

static int metal_init_activations(void *vctx, const void *config_ptr)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    const BnConfig *c = (const BnConfig *)config_ptr;
    if (!ctx || !c) return -1;

    /* Compute buffer sizes (same logic as wgpu) */
    int n_attn = (c->full_attn_interval > 0)
                     ? c->n_layers / c->full_attn_interval
                     : c->n_layers;
    int q_dim = c->n_heads * c->head_size;
    int xb_size = q_dim > c->dim ? q_dim : c->dim;

    size_t sizes[BN_GPU_BUF_COUNT] = {0};
    sizes[BN_GPU_BUF_X]           = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_BUF_XB]          = (size_t)xb_size * sizeof(float);
    sizes[BN_GPU_BUF_XB2]         = (size_t)c->dim * sizeof(float);
    sizes[BN_GPU_BUF_Q]           = (size_t)q_dim * sizeof(float);
    {
        int hb_dim = c->hidden_dim;
        if (c->moe_intermediate_size > hb_dim) hb_dim = c->moe_intermediate_size;
        sizes[BN_GPU_BUF_HB]  = (size_t)hb_dim * sizeof(float);
        sizes[BN_GPU_BUF_HB2] = (size_t)hb_dim * sizeof(float);
    }
    sizes[BN_GPU_BUF_KEY_CACHE]   = (size_t)n_attn * c->seq_len * c->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_VALUE_CACHE] = (size_t)n_attn * c->seq_len * c->kv_dim * sizeof(float);
    sizes[BN_GPU_BUF_ATT]         = (size_t)c->n_heads * c->seq_len * sizeof(float);
    sizes[BN_GPU_BUF_LOGITS]      = (size_t)c->vocab_size * sizeof(float);
    sizes[BN_GPU_BUF_ROPE_FREQ]   = (size_t)(c->head_size / 2) * sizeof(float);
    sizes[BN_GPU_BUF_SCRATCH]     = (size_t)xb_size * sizeof(float);
    {
        size_t qkv_size = (size_t)(q_dim + 2 * c->kv_dim) * sizeof(float);
        size_t gated_q_size = (size_t)(2 * q_dim) * sizeof(float);
        sizes[BN_GPU_BUF_QKV] = qkv_size > gated_q_size ? qkv_size : gated_q_size;
    }

    if (c->moe_intermediate_size > 0) {
        sizes[BN_GPU_BUF_MOE_HB]  = (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_BUF_MOE_HB2] = (size_t)c->moe_intermediate_size * sizeof(float);
        sizes[BN_GPU_BUF_MOE_OUT] = (size_t)c->dim * sizeof(float);
    }

    if (c->full_attn_interval > 0 && c->ssm_inner_size > 0) {
        int n_ssm = c->n_layers - n_attn;
        int num_v_heads = c->ssm_time_step_rank;
        int head_k_dim  = c->ssm_state_size;
        int head_v_dim  = c->ssm_inner_size / (num_v_heads > 0 ? num_v_heads : 1);
        int key_dim     = c->ssm_group_count * head_k_dim;
        int value_dim   = c->ssm_inner_size;
        int qkv_dim     = key_dim * 2 + value_dim;
        int kern        = c->ssm_conv_kernel > 0 ? c->ssm_conv_kernel : 4;

        sizes[BN_GPU_BUF_SSM_STATE]      = (size_t)n_ssm * num_v_heads * head_k_dim * head_v_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_CONV_STATE] = (size_t)n_ssm * (kern - 1) * qkv_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_QKV]        = (size_t)qkv_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_Z]          = (size_t)value_dim * sizeof(float);
        sizes[BN_GPU_BUF_SSM_ALPHA]      = (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_BUF_SSM_BETA]       = (size_t)num_v_heads * sizeof(float);
        sizes[BN_GPU_BUF_SSM_V]          = (size_t)value_dim * sizeof(float);
    }

    /* Create activation buffers (storageModeShared — unified memory) */
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        if (sizes[i] == 0) continue;
        size_t aligned = (sizes[i] + 15) & ~(size_t)15;
        ctx->act_bufs[i] = [ctx->device newBufferWithLength:aligned
                                                    options:MTLResourceStorageModeShared];
        if (!ctx->act_bufs[i]) {
            metal_free_activations(ctx);
            return -1;
        }
        ctx->act_sizes[i] = aligned;
    }

    {
        int rope_dims = c->rope_dim_count > 0 ? c->rope_dim_count : c->head_size;
        int half = rope_dims / 2;
        float *freq = (float *)malloc((size_t)half * sizeof(float));
        if (!freq) return -1;
        for (int i = 0; i < half; i++)
            freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)rope_dims);
        if (c->rope_text_dims > 0) {
            int text_pairs = c->rope_text_dims / 2;
            for (int i = text_pairs; i < half; i++)
                freq[i] = 0.0f;
        }
        memcpy([ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] contents], freq,
               (size_t)half * sizeof(float));
        free(freq);
    }

    /* Compile forward-pass shaders */
    static const struct { int id; const char *file; const char *fn; } fwd_shaders[] = {
        { BN_GPU_SHADER_RMSNORM,          "rmsnorm.metal",          "rmsnorm"          },
        { BN_GPU_SHADER_ROPE,             "rope.metal",             "rope"             },
        { BN_GPU_SHADER_GQA_SCORES,       "gqa_scores.metal",       "gqa_scores"       },
        { BN_GPU_SHADER_SOFTMAX,          "softmax.metal",          "softmax"          },
        { BN_GPU_SHADER_GQA_COMBINE,      "gqa_combine.metal",      "gqa_combine"      },
        { BN_GPU_SHADER_SILU_GATE,        "silu_gate.metal",        "silu_gate"        },
        { BN_GPU_SHADER_RELU2_GATE,       "relu2_gate.metal",       "relu2_gate"       },
        { BN_GPU_SHADER_RESIDUAL_ADD,     "residual_add.metal",     "residual_add"     },
        { BN_GPU_SHADER_BIAS_ADD,         "bias_add.metal",         "bias_add"         },
        { BN_GPU_SHADER_RESIDUAL_RMSNORM, "residual_rmsnorm.metal", "residual_rmsnorm" },
        { BN_GPU_SHADER_WEIGHTED_ADD,     "weighted_add.metal",     "weighted_add"     },
        { BN_GPU_SHADER_SSM_CONV_SILU,    "ssm_conv_silu.metal",    "ssm_conv_silu"    },
        { BN_GPU_SHADER_SSM_L2NORM,       "ssm_l2norm.metal",       "ssm_l2norm"       },
        { BN_GPU_SHADER_SSM_ALPHA_BETA,   "ssm_alpha_beta.metal",   "ssm_alpha_beta"   },
        { BN_GPU_SHADER_SSM_DELTA,        "ssm_delta.metal",        "ssm_delta"        },
        { BN_GPU_SHADER_SSM_GATE,         "ssm_gate.metal",         "ssm_gate"         },
        { BN_GPU_SHADER_PER_HEAD_RMSNORM, "per_head_rmsnorm.metal", "per_head_rmsnorm" },
        { BN_GPU_SHADER_DEINTERLEAVE_Q,   "deinterleave_q.metal",   "deinterleave_q"   },
        { BN_GPU_SHADER_SIGMOID_GATE,     "sigmoid_gate.metal",     "sigmoid_gate"     },
        { BN_GPU_SHADER_FLASH_ATTN,       "flash_attn.metal",       "flash_attn"       },
        { BN_GPU_SHADER_COPY,             "buf_copy.metal",         "buf_copy"         },
        { BN_GPU_SHADER_MATVEC_SPLIT,     "q4_matvec_split.metal",  "q4_matvec_split"  },
        { BN_GPU_SHADER_ROPE_QK,          "rope_qk.metal",          "rope_qk"          },
        { BN_GPU_SHADER_FUSED_GATEUP_SILU,"q4_fused_gateup_silu.metal","q4_fused_gateup_silu"},
        { BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT, "ssm_alpha_beta_split.metal", "ssm_alpha_beta_split" },
        { BN_GPU_SHADER_Q4K_MATVEC_SPLIT, "q4k_matvec_split.metal", "q4k_matvec_split" },
        { BN_GPU_SHADER_Q5K_MATVEC_SPLIT, "q5k_matvec_split.metal", "q5k_matvec_split" },
    };
    int n_fwd = (int)(sizeof(fwd_shaders) / sizeof(fwd_shaders[0]));
    int compiled = 0;
    for (int i = 0; i < n_fwd; i++) {
        id<MTLComputePipelineState> pso = compile_shader(ctx, ctx->shader_dir,
                                                          fwd_shaders[i].file,
                                                          fwd_shaders[i].fn);
        if (pso) {
            ctx->fwd_pipelines[fwd_shaders[i].id] = pso;
            compiled++;
        }
    }
    fprintf(stderr, "[bn:gpu:metal] compiled %d/%d forward-pass shaders\n",
            compiled, n_fwd);
    ctx->cpu_order_rmsnorm_pipeline = compile_shader(
        ctx, ctx->shader_dir, "rmsnorm_cpu_order.metal",
        "rmsnorm_cpu_order");

    ctx->ssm_prefill_rmsnorm_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_rmsnorm.metal",
        "ssm_prefill_rmsnorm");
    ctx->ssm_prefill_conv_silu_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_conv_silu.metal",
        "ssm_prefill_conv_silu");
    ctx->ssm_prefill_l2norm_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_l2norm.metal",
        "ssm_prefill_l2norm");
    ctx->ssm_prefill_alpha_beta_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_alpha_beta.metal",
        "ssm_prefill_alpha_beta");
    {
        MTLFunctionConstantValues *fc_off =
            [[MTLFunctionConstantValues alloc] init];
        ctx->ssm_prefill_delta_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "ssm_prefill_delta.metal",
            "ssm_prefill_delta", fc_off);
        MTLFunctionConstantValues *fc_on =
            [[MTLFunctionConstantValues alloc] init];
        bool fuse_ab = true;
        [fc_on setConstantValue:&fuse_ab type:MTLDataTypeBool atIndex:0];
        ctx->ssm_prefill_delta_fused_ab_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "ssm_prefill_delta.metal",
            "ssm_prefill_delta", fc_on);
        MTLFunctionConstantValues *fc_full =
            [[MTLFunctionConstantValues alloc] init];
        bool fuse_ab_full = true;
        bool fuse_l2 = true;
        [fc_full setConstantValue:&fuse_ab_full type:MTLDataTypeBool atIndex:0];
        [fc_full setConstantValue:&fuse_l2      type:MTLDataTypeBool atIndex:1];
        ctx->ssm_prefill_delta_fused_full_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "ssm_prefill_delta.metal",
            "ssm_prefill_delta", fc_full);
    }
    ctx->ssm_prefill_delta_precise_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_delta_precise.metal",
        "ssm_prefill_delta_precise");
    ctx->ssm_prefill_gate_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_gate.metal",
        "ssm_prefill_gate");
    ctx->ssm_prefill_silu_gate_stacked_pipeline = compile_shader(
        ctx, ctx->shader_dir, "ssm_prefill_silu_gate_stacked.metal",
        "ssm_prefill_silu_gate_stacked");
    {
        MTLFunctionConstantValues *empty_fc =
            [[MTLFunctionConstantValues alloc] init];
        ctx->q4k_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q4k_mul_mm.metal", "q4k_mul_mm", empty_fc);
        ctx->q5k_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q5k_mul_mm.metal", "q5k_mul_mm", empty_fc);
        ctx->q6k_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q6k_mul_mm.metal", "q6k_mul_mm", empty_fc);
        ctx->q8_0_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q8_0_mul_mm.metal", "q8_0_mul_mm", empty_fc);
        ctx->q4_0_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q4_0_mul_mm.metal", "q4_0_mul_mm", empty_fc);
        ctx->q4_1_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q4_1_mul_mm.metal", "q4_1_mul_mm", empty_fc);
        ctx->q3k_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q3k_mul_mm.metal", "q3k_mul_mm", empty_fc);
        ctx->q2k_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "q2k_mul_mm.metal", "q2k_mul_mm", empty_fc);
        ctx->f16_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "f16_mul_mm.metal", "f16_mul_mm", empty_fc);
        ctx->bf16_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "bf16_mul_mm.metal", "bf16_mul_mm", empty_fc);
        ctx->iq4_nl_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "iq4_nl_mul_mm.metal", "iq4_nl_mul_mm", empty_fc);
        ctx->iq4_xs_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "iq4_xs_mul_mm.metal", "iq4_xs_mul_mm", empty_fc);
        ctx->iq3_xxs_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "iq3_xxs_mul_mm.metal", "iq3_xxs_mul_mm", empty_fc);
        ctx->iq3_s_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "iq3_s_mul_mm.metal", "iq3_s_mul_mm", empty_fc);
        ctx->iq2_s_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "iq2_s_mul_mm.metal", "iq2_s_mul_mm", empty_fc);
        ctx->iq2_xxs_mul_mm_pipeline = compile_shader_with_fc(
            ctx, ctx->shader_dir, "iq2_xxs_mul_mm.metal", "iq2_xxs_mul_mm", empty_fc);
    }
    int n_fc_q4 = 0, n_fc_q5 = 0, n_fc_q6 = 0;
    {
        const int common_K[] = { 1024, 2048, 3584, 5504 };
        const bool bc_in = false;
        for (int i = 0; i < (int)(sizeof(common_K) / sizeof(common_K[0])); i++) {
            int K = common_K[i];
            for (int bo = 0; bo < 2; bo++) {
                bool bc_out_val = (bo != 0);
                MTLFunctionConstantValues *fc =
                    [[MTLFunctionConstantValues alloc] init];
                [fc setConstantValue:&K     type:MTLDataTypeInt  atIndex:0];
                [fc setConstantValue:&bc_in type:MTLDataTypeBool atIndex:1];
                [fc setConstantValue:&bc_out_val type:MTLDataTypeBool atIndex:2];
                id<MTLComputePipelineState> q4_pso = compile_shader_with_fc(
                    ctx, ctx->shader_dir, "q4k_mul_mm.metal", "q4k_mul_mm", fc);
                if (q4_pso) {
                    ctx->q4k_mul_mm_spec_K[n_fc_q4] = K;
                    ctx->q4k_mul_mm_spec_bc_out[n_fc_q4] = bc_out_val;
                    ctx->q4k_mul_mm_spec_pipeline[n_fc_q4] = q4_pso;
                    n_fc_q4++;
                }
                id<MTLComputePipelineState> q5_pso = compile_shader_with_fc(
                    ctx, ctx->shader_dir, "q5k_mul_mm.metal", "q5k_mul_mm", fc);
                if (q5_pso) {
                    ctx->q5k_mul_mm_spec_K[n_fc_q5] = K;
                    ctx->q5k_mul_mm_spec_bc_out[n_fc_q5] = bc_out_val;
                    ctx->q5k_mul_mm_spec_pipeline[n_fc_q5] = q5_pso;
                    n_fc_q5++;
                }
                id<MTLComputePipelineState> q6_pso = compile_shader_with_fc(
                    ctx, ctx->shader_dir, "q6k_mul_mm.metal", "q6k_mul_mm", fc);
                if (q6_pso) {
                    ctx->q6k_mul_mm_spec_K[n_fc_q6] = K;
                    ctx->q6k_mul_mm_spec_bc_out[n_fc_q6] = bc_out_val;
                    ctx->q6k_mul_mm_spec_pipeline[n_fc_q6] = q6_pso;
                    n_fc_q6++;
                }
            }
        }
    }
    (void)n_fc_q4; (void)n_fc_q5; (void)n_fc_q6;
    ctx->prefill_attn_pipeline = compile_shader(
        ctx, ctx->shader_dir, "prefill_attn.metal", "prefill_attn");
    ctx->prefill_attn_pipeline_6144 = compile_shader(
        ctx, ctx->shader_dir, "prefill_attn.metal", "prefill_attn_6144");
    ctx->prefill_kv_prep_pipeline = compile_shader(
        ctx, ctx->shader_dir, "prefill_kv_prep.metal", "prefill_kv_prep");
    ctx->ssm_prefill_enabled = 0;
    if (ctx->ssm_prefill_rmsnorm_pipeline &&
        ctx->ssm_prefill_conv_silu_pipeline &&
        ctx->ssm_prefill_l2norm_pipeline &&
        ctx->ssm_prefill_alpha_beta_pipeline &&
        ctx->ssm_prefill_delta_pipeline &&
        ctx->ssm_prefill_gate_pipeline) {
        NSUInteger lane_width =
            [ctx->ssm_prefill_delta_pipeline threadExecutionWidth];
        if (lane_width == 32) {
            ctx->ssm_prefill_enabled = 1;
        } else {
            fprintf(stderr,
                    "[bn:gpu:metal] ssm_prefill disabled: simdgroup width=%lu (need 32)\n",
                    (unsigned long)lane_width);
        }
    }

    return 0;
}

static void metal_free_activations(void *vctx)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx) return;
    for (int i = 0; i < BN_GPU_BUF_COUNT; i++) {
        ctx->act_bufs[i] = nil;
        ctx->act_sizes[i] = 0;
    }
    for (int i = 0; i < BN_GPU_SHADER_COUNT; i++)
        ctx->fwd_pipelines[i] = nil;
    ctx->ssm_prefill_rmsnorm_pipeline    = nil;
    ctx->ssm_prefill_conv_silu_pipeline  = nil;
    ctx->ssm_prefill_l2norm_pipeline     = nil;
    ctx->ssm_prefill_alpha_beta_pipeline = nil;
    ctx->ssm_prefill_delta_pipeline      = nil;
    ctx->ssm_prefill_delta_fused_full_pipeline = nil;
    ctx->ssm_prefill_delta_precise_pipeline = nil;
    ctx->ssm_prefill_gate_pipeline       = nil;
    ctx->ssm_prefill_silu_gate_stacked_pipeline = nil;
    ctx->q4k_mul_mm_pipeline = nil;
    ctx->q5k_mul_mm_pipeline = nil;
    ctx->q6k_mul_mm_pipeline = nil;
    ctx->ssm_prefill_enabled = 0;
    ctx->ssm_prefill_buf = nil;
    ctx->ssm_prefill_buf_size = 0;
}

/* ── Vtable: write/read activation ─────────────────────────────────── */

static int metal_write_activation(void *vctx, int buf_idx, const void *data,
                                   size_t size, size_t offset)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !data || buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    /* Unified memory: direct memcpy */
    memcpy((uint8_t *)[ctx->act_bufs[buf_idx] contents] + offset, data, size);
    return 0;
}

static int metal_read_activation(void *vctx, int buf_idx, void *out,
                                  size_t size, size_t offset)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !out) return -1;
    if (buf_idx == BN_GPU_DEBUG_BUF_Q8_ACT) {
        if (!ctx->q8_buf || offset + size > ctx->q8_buf_size) return -1;
        memcpy(out, (uint8_t *)[ctx->q8_buf contents] + offset, size);
        return 0;
    }
    if (buf_idx == BN_GPU_DEBUG_BUF_Q8_SCALE) {
        if (!ctx->q8_scales_buf ||
            offset + size > ctx->q8_scales_buf_size) return -1;
        memcpy(out, (uint8_t *)[ctx->q8_scales_buf contents] + offset, size);
        return 0;
    }
    if (buf_idx < 0 || buf_idx >= BN_GPU_BUF_COUNT) return -1;
    if (!ctx->act_bufs[buf_idx]) return -1;
    if (offset + size > ctx->act_sizes[buf_idx]) return -1;
    memcpy(out, (uint8_t *)[ctx->act_bufs[buf_idx] contents] + offset, size);
    return 0;
}

/* ── Vtable: matvec (standalone, not forward-pass) ─────────────────── */

static int ensure_scratch(BnMetalCtx *ctx, size_t x_need, size_t out_need)
{
    if (!ctx->x_buf || ctx->x_buf_size < x_need) {
        ctx->x_buf = [ctx->device newBufferWithLength:x_need
                                              options:MTLResourceStorageModeShared];
        if (!ctx->x_buf) return -1;
        ctx->x_buf_size = x_need;
    }
    if (!ctx->out_buf || ctx->out_buf_size < out_need) {
        ctx->out_buf = [ctx->device newBufferWithLength:out_need
                                                options:MTLResourceStorageModeShared];
        if (!ctx->out_buf) return -1;
        ctx->out_buf_size = out_need;
    }
    return 0;
}

static int ensure_q8_scratch(BnMetalCtx *ctx, int cols, int n_tokens)
{
    size_t q8_need = (size_t)cols * (size_t)n_tokens * sizeof(int8_t);
    size_t scales_need = (size_t)(cols >> 5) * (size_t)n_tokens * sizeof(float);
    if (!ctx->q8_buf || ctx->q8_buf_size < q8_need) {
        ctx->q8_buf = [ctx->device newBufferWithLength:q8_need
                                                options:MTLResourceStorageModePrivate];
        if (!ctx->q8_buf) return -1;
        ctx->q8_buf_size = q8_need;
    }
    if (!ctx->q8_scales_buf || ctx->q8_scales_buf_size < scales_need) {
        ctx->q8_scales_buf = [ctx->device newBufferWithLength:scales_need
                                                      options:MTLResourceStorageModePrivate];
        if (!ctx->q8_scales_buf) return -1;
        ctx->q8_scales_buf_size = scales_need;
    }
    return 0;
}

static int ensure_q8k_scratch(BnMetalCtx *ctx, int cols, int n_tokens)
{
    size_t q8_need = (size_t)cols * (size_t)n_tokens * sizeof(int8_t);
    size_t n_blocks = (size_t)(cols >> 8) * (size_t)n_tokens;
    size_t scales_need = n_blocks * sizeof(float);
    size_t bsums_need = n_blocks * 16 * sizeof(int16_t);
    if (!ctx->q8_buf || ctx->q8_buf_size < q8_need) {
        ctx->q8_buf = [ctx->device newBufferWithLength:q8_need
                                                options:MTLResourceStorageModePrivate];
        if (!ctx->q8_buf) return -1;
        ctx->q8_buf_size = q8_need;
    }
    if (!ctx->q8_scales_buf || ctx->q8_scales_buf_size < scales_need) {
        ctx->q8_scales_buf = [ctx->device newBufferWithLength:scales_need
                                                      options:MTLResourceStorageModePrivate];
        if (!ctx->q8_scales_buf) return -1;
        ctx->q8_scales_buf_size = scales_need;
    }
    if (!ctx->q8_bsums_buf || ctx->q8_bsums_buf_size < bsums_need) {
        ctx->q8_bsums_buf = [ctx->device newBufferWithLength:bsums_need
                                                     options:MTLResourceStorageModePrivate];
        if (!ctx->q8_bsums_buf) return -1;
        ctx->q8_bsums_buf_size = bsums_need;
    }
    return 0;
}

static void metal_encode_q8_quant(id<MTLComputeCommandEncoder> enc,
                                  BnMetalCtx *ctx,
                                  id<MTLBuffer> x_buf,
                                  uint32_t cols,
                                  uint32_t n_tokens)
{
    ctx->q8_quant_dispatches++;
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    [enc setComputePipelineState:ctx->q8_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];
    MTLSize tpg = MTLSizeMake(32, 1, 1);
    MTLSize grid = MTLSizeMake((cols + 31) / 32, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    if (ctx->q8_barriers_enabled) {
        id<MTLBuffer> bufs[2] = { ctx->q8_buf, ctx->q8_scales_buf };
        [enc memoryBarrierWithResources:bufs count:2];
    }
}

static void metal_encode_q8k_quant(id<MTLComputeCommandEncoder> enc,
                                   BnMetalCtx *ctx,
                                   id<MTLBuffer> x_buf,
                                   uint32_t cols,
                                   uint32_t n_tokens)
{
    ctx->q8k_quant_dispatches++;
    uint32_t params[8] = { cols, n_tokens, 0, 0, 0, 0, 0, 0 };
    [enc setComputePipelineState:ctx->q8k_quant_pipeline];
    [enc setBuffer:x_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
    [enc setBuffer:ctx->q8_bsums_buf offset:0 atIndex:3];
    [enc setBytes:params length:sizeof(params) atIndex:4];
    MTLSize tpg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(cols / 256, n_tokens ? n_tokens : 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    if (ctx->q8_barriers_enabled) {
        [enc memoryBarrierWithResources:&ctx->q8_buf count:1];
        id<MTLBuffer> bufs[2] = { ctx->q8_scales_buf, ctx->q8_bsums_buf };
        [enc memoryBarrierWithResources:bufs count:2];
    }
}

static int metal_matvec(void *vctx, float *out, void *W_buf, const float *x,
                         int rows, int cols, int type)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *wbuf = (BnMetalBuf *)W_buf;
    if (!ctx || !wbuf || !x || !out) return -1;
    if (type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type]) return -1;

    size_t x_size = (size_t)cols * sizeof(float);
    size_t out_size = (size_t)rows * sizeof(float);
    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;
    int use_q4_prepared_q8 = ctx->q4_q8_enabled && type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_prepared_q8_matvec_pipeline;
    int use_q4_q8 = ctx->q4_q8_enabled && type == BN_GGUF_TENSOR_Q4_0 &&
                    !wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline && ctx->q4_q8_matvec_pipeline;
    int use_q6_q8k = type == BN_GGUF_TENSOR_Q6_K &&
                     getenv("BN_METAL_ENABLE_Q6_Q8K") &&
                     ctx->q8k_quant_pipeline && ctx->q6_q8k_matvec_pipeline &&
                     (cols % 256) == 0;
    if (wbuf->q4_prepared && !use_q4_prepared_q8) return -1;
    if ((use_q4_prepared_q8 || use_q4_q8) &&
        ensure_q8_scratch(ctx, cols, 1) != 0) return -1;
    if (use_q6_q8k && ensure_q8k_scratch(ctx, cols, 1) != 0) return -1;

    memcpy([ctx->x_buf contents], x, x_size);

    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols, 1, 0, 0, 0, 0, 0 };
    if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;

    uint32_t tile_rows = (use_q4_q8 && !use_q4_prepared_q8) ? 16 : 32;
    uint32_t wg_x = ((uint32_t)rows + tile_rows - 1) / tile_rows;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        if (use_q4_prepared_q8) {
            metal_encode_q8_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->q4_prepared_q8_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
        } else if (use_q4_q8) {
            metal_encode_q8_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->q4_q8_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
        } else if (use_q6_q8k) {
            metal_encode_q8k_quant(enc, ctx, ctx->x_buf, (uint32_t)cols, 1);
            [enc setComputePipelineState:ctx->q6_q8k_matvec_pipeline];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
            [enc setBuffer:ctx->q8_bsums_buf offset:0 atIndex:3];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:4];
            [enc setBytes:params length:sizeof(params) atIndex:5];
        } else {
            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
        }

        MTLSize tpg = MTLSizeMake(tile_rows * 8, 1, 1);
        MTLSize grid = MTLSizeMake(wg_x, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, [ctx->out_buf contents], out_size);
    return 0;
}

static id<MTLComputePipelineState>
metal_mul_mm_pipeline_for(BnMetalCtx *ctx, int type,
                          int rows, int cols, int n_tokens)
{
    if (!ctx) return nil;
    if ((cols % 256) != 0) return nil;
    if (rows < 64 || n_tokens < 32) return nil;
    int prefer_fc = 1;
    bool aligned = ((rows % 64) == 0) && ((n_tokens % 32) == 0);
    if (type == BN_GGUF_TENSOR_Q4_K && ctx->q4k_mul_mm_pipeline) {
        if (prefer_fc) {
            for (int i = 0; i < 16; i++) {
                if (ctx->q4k_mul_mm_spec_K[i] == cols &&
                    ctx->q4k_mul_mm_spec_bc_out[i] == !aligned &&
                    ctx->q4k_mul_mm_spec_pipeline[i])
                    return ctx->q4k_mul_mm_spec_pipeline[i];
            }
        }
        return ctx->q4k_mul_mm_pipeline;
    }
    if (type == BN_GGUF_TENSOR_Q5_K && ctx->q5k_mul_mm_pipeline) {
        if (prefer_fc) {
            for (int i = 0; i < 16; i++) {
                if (ctx->q5k_mul_mm_spec_K[i] == cols &&
                    ctx->q5k_mul_mm_spec_bc_out[i] == !aligned &&
                    ctx->q5k_mul_mm_spec_pipeline[i])
                    return ctx->q5k_mul_mm_spec_pipeline[i];
            }
        }
        return ctx->q5k_mul_mm_pipeline;
    }
    if (type == BN_GGUF_TENSOR_Q6_K && ctx->q6k_mul_mm_pipeline) {
        if (prefer_fc) {
            for (int i = 0; i < 16; i++) {
                if (ctx->q6k_mul_mm_spec_K[i] == cols &&
                    ctx->q6k_mul_mm_spec_bc_out[i] == !aligned &&
                    ctx->q6k_mul_mm_spec_pipeline[i])
                    return ctx->q6k_mul_mm_spec_pipeline[i];
            }
        }
        return ctx->q6k_mul_mm_pipeline;
    }
    if (type == BN_GGUF_TENSOR_Q8_0   && ctx->q8_0_mul_mm_pipeline)   return ctx->q8_0_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_Q4_0   && ctx->q4_0_mul_mm_pipeline)   return ctx->q4_0_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_Q4_1   && ctx->q4_1_mul_mm_pipeline)   return ctx->q4_1_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_Q3_K   && ctx->q3k_mul_mm_pipeline)    return ctx->q3k_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_Q2_K   && ctx->q2k_mul_mm_pipeline)    return ctx->q2k_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_F16    && ctx->f16_mul_mm_pipeline)    return ctx->f16_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_BF16   && ctx->bf16_mul_mm_pipeline)   return ctx->bf16_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_IQ4_NL && ctx->iq4_nl_mul_mm_pipeline) return ctx->iq4_nl_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_IQ4_XS && ctx->iq4_xs_mul_mm_pipeline) return ctx->iq4_xs_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_IQ3_XXS && ctx->iq3_xxs_mul_mm_pipeline) return ctx->iq3_xxs_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_IQ3_S  && ctx->iq3_s_mul_mm_pipeline)  return ctx->iq3_s_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_IQ2_S  && ctx->iq2_s_mul_mm_pipeline)  return ctx->iq2_s_mul_mm_pipeline;
    if (type == BN_GGUF_TENSOR_IQ2_XXS && ctx->iq2_xxs_mul_mm_pipeline) return ctx->iq2_xxs_mul_mm_pipeline;
    return nil;
}

static void metal_encode_mul_mm(id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> pipeline,
                                 id<MTLBuffer> W_buf, size_t W_off,
                                 id<MTLBuffer> X_buf, size_t X_off,
                                 id<MTLBuffer> Y_buf, size_t Y_off,
                                 int rows, int cols, int n_tokens);

static void metal_encode_matmul(BnMetalCtx *ctx,
                                 id<MTLComputeCommandEncoder> enc,
                                 BnMetalBuf *wbuf,
                                 id<MTLBuffer> X_buf, size_t X_off,
                                 id<MTLBuffer> Y_buf, size_t Y_off,
                                 int type, int rows, int cols, int n_tokens)
{
    id<MTLComputePipelineState> mul_mm =
        metal_mul_mm_pipeline_for(ctx, type, rows, cols, n_tokens);
    if (mul_mm && wbuf->bias_offset == 0 && !wbuf->q4_prepared) {
        metal_encode_mul_mm(enc, mul_mm,
                             wbuf->buf, wbuf->offset,
                             X_buf, X_off, Y_buf, Y_off,
                             rows, cols, n_tokens);
        return;
    }
    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols,
                           (uint32_t)n_tokens, 0, 0, 0, 0, 0 };
    if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
    uint32_t tile_rows = 32;
    uint32_t wg_x = ((uint32_t)rows + tile_rows - 1) / tile_rows;
    [enc setComputePipelineState:ctx->pipelines[type]];
    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
    [enc setBuffer:X_buf offset:X_off atIndex:1];
    [enc setBuffer:Y_buf offset:Y_off atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(wg_x, (NSUInteger)n_tokens, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_mul_mm(id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> pipeline,
                                 id<MTLBuffer> W_buf, size_t W_off,
                                 id<MTLBuffer> X_buf, size_t X_off,
                                 id<MTLBuffer> Y_buf, size_t Y_off,
                                 int rows, int cols, int n_tokens)
{
    uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols,
                           (uint32_t)n_tokens, 0, 0, 0, 0, 0 };
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:W_buf offset:W_off atIndex:0];
    [enc setBuffer:X_buf offset:X_off atIndex:1];
    [enc setBuffer:Y_buf offset:Y_off atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];
    [enc setThreadgroupMemoryLength:8192 atIndex:0];
    MTLSize tpg = MTLSizeMake(128, 1, 1);
    MTLSize grid = MTLSizeMake((NSUInteger)((n_tokens + 31) / 32),
                                (NSUInteger)((rows + 63) / 64), 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
}

static int metal_matmul(void *vctx, float *out, void *W_buf, const float *X,
                         int rows, int cols, int n_tokens, int type)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *wbuf = (BnMetalBuf *)W_buf;
    if (!ctx || !wbuf || !X || !out) return -1;
    if (type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type]) return -1;
    if (wbuf->q4_prepared) return -1;

    size_t x_size = (size_t)n_tokens * cols * sizeof(float);
    size_t out_size = (size_t)n_tokens * rows * sizeof(float);
    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;

    memcpy([ctx->x_buf contents], X, x_size);

    id<MTLComputePipelineState> mul_mm =
        metal_mul_mm_pipeline_for(ctx, type, rows, cols, n_tokens);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        if (mul_mm) {
            metal_encode_mul_mm(enc, mul_mm,
                                 wbuf->buf, wbuf->offset,
                                 ctx->x_buf, 0,
                                 ctx->out_buf, 0,
                                 rows, cols, n_tokens);
        } else {
            uint32_t params[8] = { (uint32_t)rows, (uint32_t)cols,
                                   (uint32_t)n_tokens, 0, 0, 0, 0, 0 };
            uint32_t tile_rows = 32;
            uint32_t wg_x = ((uint32_t)rows + tile_rows - 1) / tile_rows;
            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, n_tokens, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, [ctx->out_buf contents], out_size);
    return 0;
}

static int metal_ensure_ssm_prefill_buf(BnMetalCtx *ctx, size_t bytes);

static int metal_dense_ffn(void *vctx, float *out,
                           void *gate_buf, void *up_buf, void *down_buf,
                           const float *x, int dim, int hidden_dim,
                           int gate_type, int up_type, int down_type,
                           int act_type)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *gate = (BnMetalBuf *)gate_buf;
    BnMetalBuf *up   = (BnMetalBuf *)up_buf;
    BnMetalBuf *down = (BnMetalBuf *)down_buf;
    if (!ctx || !out || !gate || !up || !down || !x ||
        dim <= 0 || hidden_dim <= 0 || act_type != 0)
        return -1;
    if (gate->q4_prepared || up->q4_prepared || down->q4_prepared) return -1;
    if (gate_type < 0 || gate_type >= BN_METAL_MAX_TYPES ||
        up_type   < 0 || up_type   >= BN_METAL_MAX_TYPES ||
        down_type < 0 || down_type >= BN_METAL_MAX_TYPES ||
        !ctx->pipelines[gate_type] || !ctx->pipelines[up_type] ||
        !ctx->pipelines[down_type])
        return -1;
    if (!ctx->fwd_pipelines[BN_GPU_SHADER_SILU_GATE]) return -1;

    size_t x_floats        = (size_t)dim;
    size_t hidden_floats   = (size_t)hidden_dim;
    size_t gateup_floats   = 2 * hidden_floats;
    size_t out_floats      = (size_t)dim;
    size_t total_floats    = x_floats + gateup_floats + out_floats;
    size_t off_x           = 0;
    size_t off_gateup      = x_floats;
    size_t off_out         = off_gateup + gateup_floats;

    if (metal_ensure_ssm_prefill_buf(ctx, total_floats * sizeof(float)) != 0)
        return -1;

    float *scratch = (float *)[ctx->ssm_prefill_buf contents];
    memcpy(scratch + off_x, x, x_floats * sizeof(float));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        metal_encode_matmul(ctx, enc, gate,
                             ctx->ssm_prefill_buf, off_x * sizeof(float),
                             ctx->ssm_prefill_buf, off_gateup * sizeof(float),
                             gate_type, hidden_dim, dim, 1);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        metal_encode_matmul(ctx, enc, up,
                             ctx->ssm_prefill_buf, off_x * sizeof(float),
                             ctx->ssm_prefill_buf,
                             (off_gateup + hidden_floats) * sizeof(float),
                             up_type, hidden_dim, dim, 1);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        {
            uint32_t params[8] = { (uint32_t)hidden_dim, 0, 0, 0, 0, 0, 0, 0 };
            uint32_t wg_x = ((uint32_t)hidden_dim + 255u) / 256u;
            [enc setComputePipelineState:ctx->fwd_pipelines[BN_GPU_SHADER_SILU_GATE]];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_gateup * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:(off_gateup + hidden_floats) * sizeof(float) atIndex:1];
            [enc setBytes:params length:sizeof(params) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        metal_encode_matmul(ctx, enc, down,
                             ctx->ssm_prefill_buf, off_gateup * sizeof(float),
                             ctx->ssm_prefill_buf, off_out * sizeof(float),
                             down_type, dim, hidden_dim, 1);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, scratch + off_out, out_floats * sizeof(float));
    return 0;
}

static int metal_matmul_batch(void *vctx, const BnGPUMatvecOp *ops, int n_ops,
                              const float *X, int n_tokens, int x_cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0 || !X || n_tokens <= 0 || x_cols <= 0)
        return -1;
    if (n_ops > 16) return -1;

    size_t total_out_floats = 0;
    size_t op_out_offset[16];
    for (int i = 0; i < n_ops; i++) {
        const BnGPUMatvecOp *op = &ops[i];
        BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
        if (!wbuf || op->rows <= 0 || op->cols != x_cols ||
            !op->out || op->type < 0 || op->type >= BN_METAL_MAX_TYPES)
            return -1;
        if (wbuf->q4_prepared) return -1;
        if (!ctx->pipelines[op->type]) return -1;
        op_out_offset[i] = total_out_floats;
        total_out_floats += (size_t)op->rows * (size_t)n_tokens;
    }

    size_t x_floats = (size_t)x_cols * (size_t)n_tokens;
    size_t total_floats = x_floats + total_out_floats;
    if (metal_ensure_ssm_prefill_buf(ctx, total_floats * sizeof(float)) != 0)
        return -1;

    float *scratch_base = (float *)[ctx->ssm_prefill_buf contents];
    memcpy(scratch_base, X, x_floats * sizeof(float));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        for (int i = 0; i < n_ops; i++) {
            const BnGPUMatvecOp *op = &ops[i];
            BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
            size_t out_off_floats = x_floats + op_out_offset[i];
            metal_encode_matmul(ctx, enc, wbuf,
                                 ctx->ssm_prefill_buf, 0,
                                 ctx->ssm_prefill_buf,
                                 out_off_floats * sizeof(float),
                                 op->type, op->rows, op->cols, n_tokens);
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    for (int i = 0; i < n_ops; i++) {
        const BnGPUMatvecOp *op = &ops[i];
        size_t out_off_floats = x_floats + op_out_offset[i];
        size_t out_bytes = (size_t)op->rows * (size_t)n_tokens * sizeof(float);
        memcpy(op->out, scratch_base + out_off_floats, out_bytes);
    }
    return 0;
}

static int metal_matvec_batch(void *vctx, const BnGPUMatvecOp *ops, int n_ops,
                               const float *x, int x_cols)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0 || !x) return -1;

    size_t x_size = (size_t)x_cols * sizeof(float);
    int max_rows = 0;
    for (int i = 0; i < n_ops; i++) {
        BnMetalBuf *wbuf = (BnMetalBuf *)ops[i].W_buf;
        if (wbuf && wbuf->q4_prepared)
            return -1;
        if (ops[i].rows > max_rows) max_rows = ops[i].rows;
    }
    size_t out_size = (size_t)max_rows * sizeof(float);

    if (ensure_scratch(ctx, x_size, out_size) != 0) return -1;
    memcpy([ctx->x_buf contents], x, x_size);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        for (int i = 0; i < n_ops; i++) {
            BnMetalBuf *wbuf = (BnMetalBuf *)ops[i].W_buf;
            int type = ops[i].type;
            if (!wbuf || type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type])
                continue;

            uint32_t params[8] = { (uint32_t)ops[i].rows, (uint32_t)ops[i].cols, 1, 0, 0, 0, 0, 0 };
            if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;

            uint32_t tile_rows = 32;
            uint32_t wg_x = ((uint32_t)ops[i].rows + tile_rows - 1) / tile_rows;

            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];

            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];

            /* Memory barrier between dispatches sharing out_buf */
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* Copy results to host pointers (need per-op dispatch for different out ptrs) */
    /* Re-dispatch individually since each op has a different host out pointer */
    /* TODO: optimize with a single submission + per-op output buffers */
    for (int i = 0; i < n_ops; i++) {
        BnMetalBuf *wbuf = (BnMetalBuf *)ops[i].W_buf;
        if (!wbuf) continue;
        int type = ops[i].type;
        if (type < 0 || type >= BN_METAL_MAX_TYPES || !ctx->pipelines[type]) continue;

        uint32_t params[8] = { (uint32_t)ops[i].rows, (uint32_t)ops[i].cols, 1, 0, 0, 0, 0, 0 };
        if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
        uint32_t tile_rows = 32;
        uint32_t wg_x = ((uint32_t)ops[i].rows + tile_rows - 1) / tile_rows;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:ctx->pipelines[type]];
            [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
            [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
            [enc setBuffer:ctx->out_buf offset:0 atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];

            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }

        memcpy(ops[i].out, [ctx->out_buf contents],
               (size_t)ops[i].rows * sizeof(float));
    }

    return 0;
}


static int metal_kv_cache_init(void *vctx, int n_layers, int seq_len,
                               int kv_dim)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || n_layers <= 0 || seq_len <= 0 || kv_dim <= 0) return -1;
    size_t bytes = (size_t)n_layers * (size_t)seq_len * (size_t)kv_dim *
                   sizeof(float);
    if (ctx->gpu_key_cache && ctx->gpu_kv_n_layers == n_layers &&
        ctx->gpu_kv_seq_len == seq_len && ctx->gpu_kv_dim == kv_dim)
        return 0;
    ctx->gpu_key_cache = [ctx->device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
    ctx->gpu_value_cache = [ctx->device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
    if (!ctx->gpu_key_cache || !ctx->gpu_value_cache) {
        ctx->gpu_key_cache = nil;
        ctx->gpu_value_cache = nil;
        ctx->gpu_kv_cache_bytes = 0;
        return -1;
    }
    ctx->gpu_kv_cache_bytes = bytes;
    ctx->gpu_kv_n_layers = n_layers;
    ctx->gpu_kv_seq_len = seq_len;
    ctx->gpu_kv_dim = kv_dim;
    return 0;
}

static int metal_kv_cache_write(void *vctx, int layer_idx, int pos,
                                const float *k_host, const float *v_host,
                                int n_tokens, int kv_dim)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ctx->gpu_key_cache || !ctx->gpu_value_cache) return -1;
    if (layer_idx < 0 || layer_idx >= ctx->gpu_kv_n_layers) return -1;
    if (kv_dim != ctx->gpu_kv_dim) return -1;
    if (pos < 0 || n_tokens <= 0) return -1;
    int seq_len = ctx->gpu_kv_seq_len;
    size_t layer_off_floats = (size_t)layer_idx * (size_t)seq_len * (size_t)kv_dim;
    float *k_base = (float *)[ctx->gpu_key_cache contents]   + layer_off_floats;
    float *v_base = (float *)[ctx->gpu_value_cache contents] + layer_off_floats;
    int tail = pos % seq_len;
    int first = n_tokens;
    if (tail + first > seq_len) first = seq_len - tail;
    if (first > 0) {
        memcpy(k_base + (size_t)tail * (size_t)kv_dim,
               k_host, (size_t)first * (size_t)kv_dim * sizeof(float));
        memcpy(v_base + (size_t)tail * (size_t)kv_dim,
               v_host, (size_t)first * (size_t)kv_dim * sizeof(float));
    }
    int second = n_tokens - first;
    if (second > 0) {
        memcpy(k_base,
               k_host + (size_t)first * (size_t)kv_dim,
               (size_t)second * (size_t)kv_dim * sizeof(float));
        memcpy(v_base,
               v_host + (size_t)first * (size_t)kv_dim,
               (size_t)second * (size_t)kv_dim * sizeof(float));
    }
    return 0;
}


static int metal_ensure_ssm_prefill_buf(BnMetalCtx *ctx, size_t bytes)
{
    if (ctx->ssm_prefill_buf && ctx->ssm_prefill_buf_size >= bytes)
        return 0;
    size_t aligned = (bytes + 15) & ~(size_t)15;
    ctx->ssm_prefill_buf = [ctx->device newBufferWithLength:aligned
                                                    options:MTLResourceStorageModeShared];
    if (!ctx->ssm_prefill_buf) return -1;
    ctx->ssm_prefill_buf_size = aligned;
    return 0;
}

static int metal_ensure_batch_act_buf(BnMetalCtx *ctx, size_t bytes)
{
    size_t aligned = (bytes + 15) & ~(size_t)15;
    for (int i = 0; i < 2; i++) {
        if (ctx->batch_act_buf[i] && ctx->batch_act_buf_size[i] >= aligned)
            continue;
        ctx->batch_act_buf[i] =
            [ctx->device newBufferWithLength:aligned
                                     options:MTLResourceStorageModeShared];
        if (!ctx->batch_act_buf[i]) return -1;
        ctx->batch_act_buf_size[i] = aligned;
    }
    return 0;
}

static void metal_sync_buf(BnMetalCtx *ctx,
                           id<MTLComputeCommandEncoder> __strong *encp,
                           id<MTLCommandBuffer> cmd,
                           id<MTLResource> res)
{
    (void)ctx; (void)cmd; (void)res;
    id<MTLComputeCommandEncoder> enc = *encp;
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static int metal_prefill_begin_batch(void *vctx, const float *X,
                                     int n_tokens, int dim)
{
    (void)vctx; (void)X; (void)n_tokens; (void)dim;
    return -1;
}

static int metal_prefill_flush(void *vctx, float *out, int n_tokens, int dim)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ctx->batch_active) return -1;

    [ctx->batch_enc endEncoding];
    [ctx->batch_cmd commit];
    [ctx->batch_cmd waitUntilCompleted];

    if (out && n_tokens > 0 && dim > 0) {
        size_t bytes = (size_t)n_tokens * (size_t)dim * sizeof(float);
        id<MTLBuffer> chain = ctx->batch_act_buf[ctx->batch_chain_index];
        if (chain && ctx->batch_act_buf_size[ctx->batch_chain_index] >= bytes)
            memcpy(out, [chain contents], bytes);
    }

    ctx->batch_cmd = nil;
    ctx->batch_enc = nil;
    ctx->batch_active = 0;
    return 0;
}

static int metal_prefill_ssm_layer(
        void *vctx, float *out, void *wqkv_buf, void *wz_buf,
        void *alpha_buf, void *beta_buf, void *qkvz_stacked_buf,
        void *ab_stacked_buf, void *ssm_out_buf, void *attn_norm_buf,
        void *conv1d_buf, void *dt_bias_buf, void *a_log_buf,
        void *ssm_norm_buf, void *ffn_gate_buf, void *ffn_up_buf,
        void *ffn_down_buf, void *ffn_norm_buf,
        const float *X, int n_tokens, int dim, int qkv_dim, int inner_dim,
        int num_k_heads, int head_k_dim, int num_v_heads, int head_v_dim,
        int conv_kernel, int ssm_idx, int wqkv_type, int wz_type,
        int alpha_type, int beta_type, int out_type, int hidden_dim,
        int ffn_gate_type, int ffn_up_type, int ffn_down_type, int act_type,
        float norm_eps, int *did_ffn)
{
    (void)qkvz_stacked_buf; (void)ab_stacked_buf;

    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (did_ffn) *did_ffn = 0;
    if (!ctx) return -1;
    if (!ctx->ssm_prefill_enabled) return -1;

    if (n_tokens <= 1 || dim <= 0 || qkv_dim <= 0 || inner_dim <= 0 ||
        num_k_heads <= 0 || num_v_heads <= 0 ||
        head_k_dim != 128 || head_v_dim != 128 ||
        inner_dim != num_v_heads * head_v_dim ||
        qkv_dim != 2 * num_k_heads * head_k_dim + inner_dim ||
        conv_kernel < 2 || conv_kernel > 8 || ssm_idx < 0)
        return -1;

    BnMetalBuf *wqkv      = (BnMetalBuf *)wqkv_buf;
    BnMetalBuf *wz        = (BnMetalBuf *)wz_buf;
    BnMetalBuf *alpha_w   = (BnMetalBuf *)alpha_buf;
    BnMetalBuf *beta_w    = (BnMetalBuf *)beta_buf;
    BnMetalBuf *ssm_out   = (BnMetalBuf *)ssm_out_buf;
    BnMetalBuf *attn_norm = (BnMetalBuf *)attn_norm_buf;
    BnMetalBuf *conv1d    = (BnMetalBuf *)conv1d_buf;
    BnMetalBuf *dt_bias   = (BnMetalBuf *)dt_bias_buf;
    BnMetalBuf *a_log     = (BnMetalBuf *)a_log_buf;
    BnMetalBuf *ssm_norm  = (BnMetalBuf *)ssm_norm_buf;
    if (!wqkv || !wz || !alpha_w || !beta_w || !ssm_out ||
        !attn_norm || !conv1d || !dt_bias || !a_log || !ssm_norm)
        return -1;
    if (wqkv->q4_prepared || wz->q4_prepared || alpha_w->q4_prepared ||
        beta_w->q4_prepared || ssm_out->q4_prepared)
        return -1;
    if (wqkv_type < 0 || wqkv_type >= BN_METAL_MAX_TYPES ||
        wz_type   < 0 || wz_type   >= BN_METAL_MAX_TYPES ||
        alpha_type < 0 || alpha_type >= BN_METAL_MAX_TYPES ||
        beta_type  < 0 || beta_type  >= BN_METAL_MAX_TYPES ||
        out_type   < 0 || out_type   >= BN_METAL_MAX_TYPES ||
        !ctx->pipelines[wqkv_type] || !ctx->pipelines[wz_type] ||
        !ctx->pipelines[alpha_type] || !ctx->pipelines[beta_type] ||
        !ctx->pipelines[out_type])
        return -1;
    if (!ctx->act_bufs[BN_GPU_BUF_SSM_STATE] ||
        !ctx->act_bufs[BN_GPU_BUF_SSM_CONV_STATE])
        return -1;

    BnMetalBuf *ffn_gate_w = (BnMetalBuf *)ffn_gate_buf;
    BnMetalBuf *ffn_up_w   = (BnMetalBuf *)ffn_up_buf;
    BnMetalBuf *ffn_down_w = (BnMetalBuf *)ffn_down_buf;
    BnMetalBuf *ffn_norm_w = (BnMetalBuf *)ffn_norm_buf;
    int ffn_stacked = (ffn_gate_w && !ffn_up_w);
    int fuse_ffn = (hidden_dim > 0 && act_type == 0 &&
                    ffn_gate_w && ffn_gate_w->buf &&
                    (ffn_stacked || (ffn_up_w && ffn_up_w->buf &&
                                     !ffn_up_w->q4_prepared)) &&
                    ffn_down_w && ffn_down_w->buf &&
                    ffn_norm_w && ffn_norm_w->buf &&
                    !ffn_gate_w->q4_prepared &&
                    !ffn_down_w->q4_prepared &&
                    ffn_gate_type >= 0 && ffn_gate_type < BN_METAL_MAX_TYPES &&
                    ffn_down_type >= 0 && ffn_down_type < BN_METAL_MAX_TYPES &&
                    ctx->pipelines[ffn_gate_type] &&
                    ctx->pipelines[ffn_down_type] &&
                    ctx->fwd_pipelines[BN_GPU_SHADER_SILU_GATE] &&
                    ctx->fwd_pipelines[BN_GPU_SHADER_COPY] &&
                    (!ffn_stacked || ctx->ssm_prefill_silu_gate_stacked_pipeline) &&
                    (ffn_stacked || (ffn_up_type >= 0 &&
                                     ffn_up_type < BN_METAL_MAX_TYPES &&
                                     ctx->pipelines[ffn_up_type])));

    size_t dim_values   = (size_t)n_tokens * (size_t)dim;
    size_t qkv_values   = (size_t)n_tokens * (size_t)qkv_dim;
    size_t z_values     = (size_t)n_tokens * (size_t)inner_dim;
    size_t ab_values    = (size_t)n_tokens * (size_t)num_v_heads;
    size_t hidden_values = fuse_ffn
        ? (size_t)n_tokens * (size_t)hidden_dim : 0;
    size_t off_norm  = 0;
    size_t off_qkv   = off_norm  + dim_values;
    size_t off_z     = off_qkv   + qkv_values;
    size_t off_ssm   = off_z     + z_values;
    size_t off_alpha = off_ssm   + z_values;
    size_t off_beta  = off_alpha + ab_values;
    size_t off_ffn_residual = off_beta + ab_values;
    size_t off_ffn_norm     = off_ffn_residual + (fuse_ffn ? dim_values : 0);
    size_t off_ffn_act      = off_ffn_norm     + (fuse_ffn ? dim_values : 0);
    size_t off_ffn_post     = off_ffn_act + hidden_values * 2;
    size_t scratch_floats   = off_ffn_post + hidden_values;

    if (metal_ensure_ssm_prefill_buf(ctx, scratch_floats * sizeof(float)) != 0)
        return -1;

    id<MTLBuffer> x_id  = nil;
    id<MTLBuffer> out_id = nil;
    if (ctx->batch_active) {
        if (metal_ensure_batch_act_buf(ctx, dim_values * sizeof(float)) != 0)
            return -1;
        x_id  = ctx->batch_act_buf[ctx->batch_chain_index];
        out_id = ctx->batch_act_buf[ctx->batch_chain_index ^ 1];
    } else {
        if (ensure_scratch(ctx, dim_values * sizeof(float),
                           dim_values * sizeof(float)) != 0)
            return -1;
        x_id  = ctx->x_buf;
        out_id = ctx->out_buf;
        memcpy([x_id contents], X, dim_values * sizeof(float));
    }

    const uint32_t key_dim = (uint32_t)(num_k_heads * head_k_dim);
    const float q_scale = 1.0f / sqrtf((float)head_k_dim);
    const uint32_t state_off_bytes =
        (uint32_t)((size_t)ssm_idx * (size_t)num_v_heads *
                   (size_t)head_k_dim * (size_t)head_v_dim * sizeof(float));
    const uint32_t conv_off_floats =
        (uint32_t)((size_t)ssm_idx * (size_t)(conv_kernel - 1) *
                   (size_t)qkv_dim);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd =
            ctx->batch_active ? ctx->batch_cmd : [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            ctx->batch_active ? ctx->batch_enc : [cmd computeCommandEncoder];

#define SSM_CHECKPOINT(NAME, SLOT) do { (void)(NAME); (void)(SLOT); } while (0)

        {
            uint32_t params[8] = { (uint32_t)dim, 0, 0, 0, 0, 0, 0, 0 };
            memcpy(&params[1], &norm_eps, sizeof(float));
            [enc setComputePipelineState:ctx->ssm_prefill_rmsnorm_pipeline];
            [enc setBuffer:x_id offset:0 atIndex:0];
            [enc setBuffer:attn_norm->buf offset:attn_norm->offset atIndex:1];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_norm * sizeof(float) atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("rmsnorm_in", 0);

        struct {
            BnMetalBuf *w;
            int type;
            uint32_t rows;
            size_t   out_off;
        } projs[4] = {
            { wqkv,    wqkv_type,  (uint32_t)qkv_dim,     off_qkv   },
            { wz,      wz_type,    (uint32_t)inner_dim,   off_z     },
            { alpha_w, alpha_type, (uint32_t)num_v_heads, off_alpha },
            { beta_w,  beta_type,  (uint32_t)num_v_heads, off_beta  },
        };
        for (int proj_i = 0; proj_i < 4; proj_i++) {
            metal_encode_matmul(ctx, enc, projs[proj_i].w,
                                 ctx->ssm_prefill_buf, off_norm * sizeof(float),
                                 ctx->ssm_prefill_buf,
                                 projs[proj_i].out_off * sizeof(float),
                                 projs[proj_i].type,
                                 (int)projs[proj_i].rows, dim, n_tokens);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            SSM_CHECKPOINT(proj_i == 0 ? "wqkv_matmul"
                         : proj_i == 1 ? "wz_matmul"
                         : proj_i == 2 ? "alpha_matmul"
                         :               "beta_matmul",
                           1 + proj_i);
        }

        {
            uint32_t params[8] = { (uint32_t)qkv_dim, (uint32_t)conv_kernel,
                                   conv_off_floats, (uint32_t)n_tokens,
                                   0, 0, 0, 0 };
            uint32_t wg_x = ((uint32_t)qkv_dim + 255u) / 256u;
            [enc setComputePipelineState:ctx->ssm_prefill_conv_silu_pipeline];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_qkv * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_CONV_STATE]
                    offset:0 atIndex:1];
            [enc setBuffer:conv1d->buf offset:conv1d->offset atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("conv_silu", 5);

        int use_precise = 0;
        int fuse_alpha_beta =
            !use_precise &&
            ctx->ssm_prefill_delta_fused_ab_pipeline != nil;
        int fuse_l2norm = 0;

        if (!fuse_l2norm) {
            uint32_t params[8] = { (uint32_t)head_k_dim, 0, key_dim,
                                   (uint32_t)num_k_heads, (uint32_t)qkv_dim,
                                   0, 0, 0 };
            [enc setComputePipelineState:ctx->ssm_prefill_l2norm_pipeline];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_qkv * sizeof(float) atIndex:0];
            [enc setBytes:params length:sizeof(params) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_k_heads,
                                                  (NSUInteger)n_tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("l2norm", 6);

        if (!fuse_alpha_beta) {
            uint32_t params[8] = { (uint32_t)num_v_heads, (uint32_t)n_tokens,
                                   0, 0, 0, 0, 0, 0 };
            uint32_t total = (uint32_t)num_v_heads * (uint32_t)n_tokens;
            uint32_t wg_x = (total + 255u) / 256u;
            [enc setComputePipelineState:ctx->ssm_prefill_alpha_beta_pipeline];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_alpha * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_beta * sizeof(float) atIndex:1];
            [enc setBuffer:dt_bias->buf offset:dt_bias->offset atIndex:2];
            [enc setBuffer:a_log->buf offset:a_log->offset atIndex:3];
            [enc setBytes:params length:sizeof(params) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("alpha_beta", 7);

        {
            uint32_t qs_bits;
            memcpy(&qs_bits, &q_scale, sizeof(float));
            uint32_t params[8] = { (uint32_t)n_tokens, (uint32_t)qkv_dim,
                                   (uint32_t)num_k_heads,
                                   (uint32_t)num_v_heads, qs_bits,
                                   state_off_bytes, 0, key_dim };
            id<MTLComputePipelineState> delta_pso = use_precise
                ? ctx->ssm_prefill_delta_precise_pipeline
                : (fuse_l2norm
                       ? ctx->ssm_prefill_delta_fused_full_pipeline
                       : (fuse_alpha_beta
                              ? ctx->ssm_prefill_delta_fused_ab_pipeline
                              : ctx->ssm_prefill_delta_pipeline));
            [enc setComputePipelineState:delta_pso];
            [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_STATE] offset:0 atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_ssm * sizeof(float) atIndex:1];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_qkv * sizeof(float) atIndex:2];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_alpha * sizeof(float) atIndex:3];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_beta * sizeof(float) atIndex:4];
            [enc setBytes:params length:sizeof(params) atIndex:5];
            [enc setBuffer:dt_bias->buf offset:dt_bias->offset atIndex:6];
            [enc setBuffer:a_log->buf offset:a_log->offset atIndex:7];
            if (use_precise) {
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_v_heads,
                                                       1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_v_heads,
                                                       32, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("delta", 8);

        {
            uint32_t eps_bits;
            memcpy(&eps_bits, &norm_eps, sizeof(float));
            uint32_t params[8] = { (uint32_t)head_v_dim, eps_bits,
                                   (uint32_t)num_v_heads, 0, 0, 0, 0, 0 };
            [enc setComputePipelineState:ctx->ssm_prefill_gate_pipeline];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_ssm * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf
                    offset:off_z * sizeof(float) atIndex:1];
            [enc setBuffer:ssm_norm->buf offset:ssm_norm->offset atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_v_heads,
                                                   (NSUInteger)n_tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("gate", 9);

        {
            metal_encode_matmul(ctx, enc, ssm_out,
                                 ctx->ssm_prefill_buf, off_ssm * sizeof(float),
                                 out_id, 0,
                                 out_type, dim, inner_dim, n_tokens);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("ssm_out_matmul", 10);

        {
            uint32_t total = (uint32_t)dim_values;
            uint32_t params[8] = { total, 0, 0, 0, 0, 0, 0, 0 };
            uint32_t wg_x = (total + 255u) / 256u;
            [enc setComputePipelineState:ctx->fwd_pipelines[BN_GPU_SHADER_RESIDUAL_ADD]];
            [enc setBuffer:out_id offset:0 atIndex:0];
            [enc setBuffer:x_id offset:0 atIndex:1];
            [enc setBytes:params length:sizeof(params) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        SSM_CHECKPOINT("residual_add", 11);

        if (fuse_ffn) {
            {
                uint32_t params[8] = { (uint32_t)dim, 0, 0, 0, 0, 0, 0, 0 };
                memcpy(&params[1], &norm_eps, sizeof(float));
                [enc setComputePipelineState:ctx->ssm_prefill_rmsnorm_pipeline];
                [enc setBuffer:out_id offset:0 atIndex:0];
                [enc setBuffer:ffn_norm_w->buf offset:ffn_norm_w->offset atIndex:1];
                [enc setBuffer:ctx->ssm_prefill_buf
                        offset:off_ffn_norm * sizeof(float) atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            SSM_CHECKPOINT("ffn_rmsnorm", 12);

            metal_sync_buf(ctx, &enc, cmd, ctx->ssm_prefill_buf);
            if (ffn_stacked) {
                metal_encode_matmul(ctx, enc, ffn_gate_w,
                                     ctx->ssm_prefill_buf,
                                     off_ffn_norm * sizeof(float),
                                     ctx->ssm_prefill_buf,
                                     off_ffn_act * sizeof(float),
                                     ffn_gate_type, hidden_dim * 2,
                                     dim, n_tokens);
                metal_sync_buf(ctx, &enc, cmd, ctx->ssm_prefill_buf);
                SSM_CHECKPOINT("ffn_gateup_stacked", 13);
            } else {
                struct {
                    BnMetalBuf *w;
                    int type;
                    size_t out_off;
                } ffn_projs[2] = {
                    { ffn_gate_w, ffn_gate_type, off_ffn_act },
                    { ffn_up_w,   ffn_up_type,   off_ffn_act + hidden_values },
                };
                for (int proj_i = 0; proj_i < 2; proj_i++) {
                    metal_encode_matmul(ctx, enc, ffn_projs[proj_i].w,
                                         ctx->ssm_prefill_buf,
                                         off_ffn_norm * sizeof(float),
                                         ctx->ssm_prefill_buf,
                                         ffn_projs[proj_i].out_off * sizeof(float),
                                         ffn_projs[proj_i].type, hidden_dim,
                                         dim, n_tokens);
                    metal_sync_buf(ctx, &enc, cmd, ctx->ssm_prefill_buf);
                }
            }

            if (ffn_stacked) {
                uint32_t params[8] = { (uint32_t)hidden_dim,
                                       (uint32_t)n_tokens, 0, 0, 0, 0, 0, 0 };
                uint32_t wg_x = ((uint32_t)hidden_dim + 255u) / 256u;
                [enc setComputePipelineState:ctx->ssm_prefill_silu_gate_stacked_pipeline];
                [enc setBuffer:ctx->ssm_prefill_buf
                        offset:off_ffn_act * sizeof(float) atIndex:0];
                [enc setBuffer:ctx->ssm_prefill_buf
                        offset:off_ffn_post * sizeof(float) atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(wg_x, (NSUInteger)n_tokens, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                uint32_t total = (uint32_t)hidden_values;
                uint32_t params[8] = { total, 0, 0, 0, 0, 0, 0, 0 };
                uint32_t wg_x = (total + 255u) / 256u;
                [enc setComputePipelineState:ctx->fwd_pipelines[BN_GPU_SHADER_SILU_GATE]];
                [enc setBuffer:ctx->ssm_prefill_buf
                        offset:off_ffn_act * sizeof(float) atIndex:0];
                [enc setBuffer:ctx->ssm_prefill_buf
                        offset:(off_ffn_act + hidden_values) * sizeof(float) atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            {
                size_t ffn_down_in_off =
                    ffn_stacked ? off_ffn_post : off_ffn_act;
                metal_encode_matmul(ctx, enc, ffn_down_w,
                                     ctx->ssm_prefill_buf,
                                     ffn_down_in_off * sizeof(float),
                                     ctx->ssm_prefill_buf,
                                     off_ffn_residual * sizeof(float),
                                     ffn_down_type, dim, hidden_dim, n_tokens);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            SSM_CHECKPOINT("ffn_down_matmul", 14);

            {
                uint32_t total = (uint32_t)dim_values;
                uint32_t params[8] = { total, 0, 0, 0, 0, 0, 0, 0 };
                uint32_t wg_x = (total + 255u) / 256u;
                [enc setComputePipelineState:ctx->fwd_pipelines[BN_GPU_SHADER_RESIDUAL_ADD]];
                [enc setBuffer:out_id offset:0 atIndex:0];
                [enc setBuffer:ctx->ssm_prefill_buf
                        offset:off_ffn_residual * sizeof(float) atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        if (ctx->batch_active) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ctx->batch_chain_index ^= 1;
        } else {
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }
#undef SSM_CHECKPOINT

    if (out && !ctx->batch_active)
        memcpy(out, [ctx->out_buf contents], dim_values * sizeof(float));

    if (did_ffn) *did_ffn = fuse_ffn ? 1 : 0;

    return 0;
}


static int metal_ensure_attn_scratch(BnMetalCtx *ctx,
                                     size_t q_bytes,
                                     size_t kv_bytes,
                                     size_t out_bytes)
{
    size_t need = q_bytes;
    if (kv_bytes > need)  need = kv_bytes;
    if (out_bytes > need) need = out_bytes;
    size_t aligned = (need + 15) & ~(size_t)15;
    if (!ctx->ssm_prefill_buf || ctx->ssm_prefill_buf_size < aligned)
        return metal_ensure_ssm_prefill_buf(ctx, aligned);
    return 0;
}

static int metal_prefill_attention(void *vctx, float *out,
                                   const float *Q, const float *K,
                                   const float *V,
                                   int n_tokens, int n_heads,
                                   int n_kv_heads, int head_size,
                                   int kv_mul, int kv_dim,
                                   float attention_scale,
                                   int pos0, int seq_len,
                                   const float *key_cache,
                                   const float *value_cache, size_t loff)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !out || !Q || !K || !V) return -1;
    if (n_tokens <= 1 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_size <= 0 || kv_mul <= 0 || kv_dim <= 0)
        return -1;
    size_t max_pos_n = (size_t)pos0 + (size_t)n_tokens;
    id<MTLComputePipelineState> attn_pso = NULL;
    if (max_pos_n <= 4096)
        attn_pso = ctx->prefill_attn_pipeline;
    else if (max_pos_n <= 6144)
        attn_pso = ctx->prefill_attn_pipeline_6144
                     ?: ctx->prefill_attn_pipeline;
    else
        return -1;
    if (!attn_pso) return -1;

    int use_cache = pos0 > 0 ? 1 : 0;
    if (use_cache && (!key_cache || !value_cache || seq_len <= 0))
        return -1;
    if (use_cache && (size_t)pos0 + (size_t)loff > SIZE_MAX / sizeof(float))
        return -1;

    int gpu_cache_path = 0;
    size_t gpu_layer_loff = 0;
    if (use_cache && ctx->gpu_key_cache && ctx->gpu_value_cache &&
        ctx->gpu_kv_dim == kv_dim &&
        (size_t)pos0 <= (size_t)ctx->gpu_kv_seq_len) {
        size_t layer_stride =
            (size_t)ctx->gpu_kv_seq_len * (size_t)kv_dim;
        if (layer_stride > 0 && (loff % layer_stride) == 0) {
            size_t layer_idx = loff / layer_stride;
            if (layer_idx < (size_t)ctx->gpu_kv_n_layers) {
                gpu_cache_path = 1;
                gpu_layer_loff = layer_idx * layer_stride;
            }
        }
    }

    size_t q_floats   = (size_t)n_tokens * (size_t)n_heads    * (size_t)head_size;
    size_t kv_floats  = (size_t)n_tokens * (size_t)n_kv_heads * (size_t)head_size;
    size_t out_floats = q_floats;
    size_t past_floats = (use_cache && !gpu_cache_path)
                           ? (size_t)pos0 * (size_t)kv_dim : 0;
    size_t total = q_floats + kv_floats + kv_floats + out_floats +
                   past_floats * 2;
    if (metal_ensure_ssm_prefill_buf(ctx, total * sizeof(float)) != 0)
        return -1;
    (void)metal_ensure_attn_scratch;
    size_t off_q       = 0;
    size_t off_k       = off_q + q_floats;
    size_t off_v       = off_k + kv_floats;
    size_t off_o       = off_v + kv_floats;
    size_t off_past_k  = off_o + out_floats;
    size_t off_past_v  = off_past_k + past_floats;

    float *scratch_base = (float *)[ctx->ssm_prefill_buf contents];
    memcpy(scratch_base + off_q, Q, q_floats * sizeof(float));
    memcpy(scratch_base + off_k, K, kv_floats * sizeof(float));
    memcpy(scratch_base + off_v, V, kv_floats * sizeof(float));
    if (use_cache && !gpu_cache_path) {
        if ((size_t)pos0 > (size_t)seq_len)
            return -1;
        memcpy(scratch_base + off_past_k,
               key_cache   + loff, past_floats * sizeof(float));
        memcpy(scratch_base + off_past_v,
               value_cache + loff, past_floats * sizeof(float));
    }

    uint32_t scale_bits;
    memcpy(&scale_bits, &attention_scale, sizeof(float));
    uint32_t shader_seq_len = gpu_cache_path ? (uint32_t)ctx->gpu_kv_seq_len
                            : (use_cache ? (uint32_t)pos0 : 0u);
    uint32_t params[12] = {
        (uint32_t)n_tokens, (uint32_t)n_heads, (uint32_t)n_kv_heads,
        (uint32_t)head_size, (uint32_t)kv_mul,
        (uint32_t)(use_cache ? pos0 : 0),
        shader_seq_len, scale_bits,
        0u,
        (uint32_t)kv_dim, (uint32_t)use_cache,
        0u };

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:attn_pso];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_q       * sizeof(float) atIndex:0];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_k       * sizeof(float) atIndex:1];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_v       * sizeof(float) atIndex:2];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_o       * sizeof(float) atIndex:3];
        if (gpu_cache_path) {
            [enc setBuffer:ctx->gpu_key_cache
                    offset:gpu_layer_loff * sizeof(float) atIndex:4];
            [enc setBuffer:ctx->gpu_value_cache
                    offset:gpu_layer_loff * sizeof(float) atIndex:5];
        } else {
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_past_k * sizeof(float) atIndex:4];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_past_v * sizeof(float) atIndex:5];
        }
        [enc setBytes:params length:sizeof(params) atIndex:6];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_q * sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads,
                                              (NSUInteger)n_tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, scratch_base + off_o, out_floats * sizeof(float));
    return 0;
}


static int metal_prefill_kv_prep(void *vctx, float *K, float *V,
                                 const float *k_bias, const float *v_bias,
                                 const float *k_norm_w,
                                 const float *rope_cos, const float *rope_sin,
                                 int n_tokens, int n_kv_heads, int head_size,
                                 int rope_dims, int qk_norm_per_head,
                                 float norm_eps)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !K || !V) return -1;
    if (n_tokens <= 0 || n_kv_heads <= 0 || head_size <= 0) return -1;
    if (head_size > 256) return -1;
    if (!ctx->prefill_kv_prep_pipeline) return -1;

    int use_k_bias = k_bias != NULL ? 1 : 0;
    int use_v_bias = v_bias != NULL ? 1 : 0;
    int use_k_norm = k_norm_w != NULL ? 1 : 0;
    int use_rope   = (rope_cos != NULL && rope_sin != NULL && rope_dims > 0) ? 1 : 0;

    size_t kv_dim = (size_t)n_kv_heads * (size_t)head_size;
    size_t kv_floats = (size_t)n_tokens * kv_dim;
    size_t norm_floats = (size_t)(qk_norm_per_head ? n_kv_heads : 1) * head_size;
    size_t bias_floats = kv_dim;
    size_t rope_floats = (size_t)n_tokens * (size_t)(rope_dims / 2);
    size_t off_k     = 0;
    size_t off_v     = off_k    + kv_floats;
    size_t off_kb    = off_v    + kv_floats;
    size_t off_vb    = off_kb   + (use_k_bias ? bias_floats : 0);
    size_t off_kn    = off_vb   + (use_v_bias ? bias_floats : 0);
    size_t off_rc    = off_kn   + (use_k_norm ? norm_floats : 0);
    size_t off_rs    = off_rc   + (use_rope ? rope_floats : 0);
    size_t total     = off_rs   + (use_rope ? rope_floats : 0);
    if (metal_ensure_ssm_prefill_buf(ctx, total * sizeof(float)) != 0)
        return -1;

    float *scratch = (float *)[ctx->ssm_prefill_buf contents];
    memcpy(scratch + off_k, K, kv_floats * sizeof(float));
    memcpy(scratch + off_v, V, kv_floats * sizeof(float));
    if (use_k_bias) memcpy(scratch + off_kb, k_bias, bias_floats * sizeof(float));
    if (use_v_bias) memcpy(scratch + off_vb, v_bias, bias_floats * sizeof(float));
    if (use_k_norm) memcpy(scratch + off_kn, k_norm_w, norm_floats * sizeof(float));
    if (use_rope) {
        memcpy(scratch + off_rc, rope_cos, rope_floats * sizeof(float));
        memcpy(scratch + off_rs, rope_sin, rope_floats * sizeof(float));
    }

    uint32_t eps_bits;
    memcpy(&eps_bits, &norm_eps, sizeof(float));
    uint32_t params[12] = {
        (uint32_t)n_kv_heads, (uint32_t)head_size, (uint32_t)n_tokens,
        (uint32_t)rope_dims, (uint32_t)qk_norm_per_head,
        (uint32_t)use_k_bias, (uint32_t)use_v_bias,
        (uint32_t)use_k_norm, (uint32_t)use_rope,
        eps_bits, (uint32_t)(rope_dims / 2), 0u };

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->prefill_kv_prep_pipeline];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_k  * sizeof(float) atIndex:0];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_v  * sizeof(float) atIndex:1];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_kb * sizeof(float) atIndex:2];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_vb * sizeof(float) atIndex:3];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_kn * sizeof(float) atIndex:4];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_rc * sizeof(float) atIndex:5];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_rs * sizeof(float) atIndex:6];
        [enc setBytes:params length:sizeof(params) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_kv_heads,
                                              (NSUInteger)n_tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(K, scratch + off_k, kv_floats * sizeof(float));
    memcpy(V, scratch + off_v, kv_floats * sizeof(float));
    return 0;
}


static int metal_prefill_attention_wo(void *vctx, float *out, void *wo_buf,
                                      const float *Q, const float *K,
                                      const float *V, int n_tokens,
                                      int n_heads, int n_kv_heads,
                                      int head_size, int kv_mul, int kv_dim,
                                      int wo_rows, int wo_cols, int wo_type,
                                      float attention_scale,
                                      int pos0, int seq_len,
                                      const float *key_cache,
                                      const float *value_cache, size_t loff,
                                      const float *gate)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    BnMetalBuf *wo  = (BnMetalBuf *)wo_buf;
    if (!ctx || !out || !wo || !Q || !K || !V) return -1;
    if (n_tokens <= 1 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_size <= 0 || kv_mul <= 0 || kv_dim <= 0 ||
        wo_rows <= 0 || wo_cols <= 0)
        return -1;
    if (wo_type < 0 || wo_type >= BN_METAL_MAX_TYPES) return -1;
    if (wo->q4_prepared) return -1;
    if (!ctx->pipelines[wo_type] && !ctx->q4k_mul_mm_pipeline &&
        !ctx->q5k_mul_mm_pipeline)
        return -1;
    size_t max_pos_n_wo = (size_t)pos0 + (size_t)n_tokens;
    id<MTLComputePipelineState> attn_pso_wo = NULL;
    if (max_pos_n_wo <= 4096)
        attn_pso_wo = ctx->prefill_attn_pipeline;
    else if (max_pos_n_wo <= 6144)
        attn_pso_wo = ctx->prefill_attn_pipeline_6144
                        ?: ctx->prefill_attn_pipeline;
    else
        return -1;
    if (!attn_pso_wo) return -1;
    if (wo_cols != n_heads * head_size) return -1;

    int use_cache = pos0 > 0 ? 1 : 0;
    int use_q_gate = gate != NULL ? 1 : 0;
    if (use_cache && (!key_cache || !value_cache || seq_len <= 0))
        return -1;

    int gpu_cache_path = 0;
    size_t gpu_layer_loff = 0;
    if (use_cache && ctx->gpu_key_cache && ctx->gpu_value_cache &&
        ctx->gpu_kv_dim == kv_dim &&
        (size_t)pos0 <= (size_t)ctx->gpu_kv_seq_len) {
        size_t layer_stride =
            (size_t)ctx->gpu_kv_seq_len * (size_t)kv_dim;
        if (layer_stride > 0 && (loff % layer_stride) == 0) {
            size_t layer_idx = loff / layer_stride;
            if (layer_idx < (size_t)ctx->gpu_kv_n_layers) {
                gpu_cache_path = 1;
                gpu_layer_loff = layer_idx * layer_stride;
            }
        }
    }

    size_t q_floats    = (size_t)n_tokens * (size_t)n_heads * (size_t)head_size;
    size_t kv_floats   = (size_t)n_tokens * (size_t)kv_dim;
    size_t out_floats  = q_floats;
    size_t wo_floats   = (size_t)n_tokens * (size_t)wo_rows;
    size_t past_floats = (use_cache && !gpu_cache_path)
                           ? (size_t)pos0 * (size_t)kv_dim : 0;
    size_t gate_floats = use_q_gate ? q_floats : 0;
    size_t total = q_floats + kv_floats + kv_floats + out_floats + wo_floats +
                   past_floats * 2 + gate_floats;
    if (metal_ensure_ssm_prefill_buf(ctx, total * sizeof(float)) != 0)
        return -1;
    size_t off_q        = 0;
    size_t off_k        = off_q        + q_floats;
    size_t off_v        = off_k        + kv_floats;
    size_t off_attn_out = off_v        + kv_floats;
    size_t off_wo_out   = off_attn_out + out_floats;
    size_t off_past_k   = off_wo_out   + wo_floats;
    size_t off_past_v   = off_past_k   + past_floats;
    size_t off_gate     = off_past_v   + past_floats;

    float *scratch_base = (float *)[ctx->ssm_prefill_buf contents];
    memcpy(scratch_base + off_q, Q, q_floats * sizeof(float));
    memcpy(scratch_base + off_k, K, kv_floats * sizeof(float));
    memcpy(scratch_base + off_v, V, kv_floats * sizeof(float));
    if (use_cache && !gpu_cache_path) {
        if ((size_t)pos0 > (size_t)seq_len) return -1;
        memcpy(scratch_base + off_past_k, key_cache + loff,
               past_floats * sizeof(float));
        memcpy(scratch_base + off_past_v, value_cache + loff,
               past_floats * sizeof(float));
    }
    if (use_q_gate)
        memcpy(scratch_base + off_gate, gate, gate_floats * sizeof(float));

    uint32_t scale_bits;
    memcpy(&scale_bits, &attention_scale, sizeof(float));
    uint32_t shader_seq_len = gpu_cache_path ? (uint32_t)ctx->gpu_kv_seq_len
                            : (use_cache ? (uint32_t)pos0 : 0u);
    uint32_t params[12] = {
        (uint32_t)n_tokens, (uint32_t)n_heads, (uint32_t)n_kv_heads,
        (uint32_t)head_size, (uint32_t)kv_mul,
        (uint32_t)(use_cache ? pos0 : 0),
        shader_seq_len, scale_bits,
        0u, (uint32_t)kv_dim, (uint32_t)use_cache, (uint32_t)use_q_gate };

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:attn_pso_wo];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_q        * sizeof(float) atIndex:0];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_k        * sizeof(float) atIndex:1];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_v        * sizeof(float) atIndex:2];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_attn_out * sizeof(float) atIndex:3];
        if (gpu_cache_path) {
            [enc setBuffer:ctx->gpu_key_cache
                    offset:gpu_layer_loff * sizeof(float) atIndex:4];
            [enc setBuffer:ctx->gpu_value_cache
                    offset:gpu_layer_loff * sizeof(float) atIndex:5];
        } else {
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_past_k * sizeof(float) atIndex:4];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_past_v * sizeof(float) atIndex:5];
        }
        [enc setBytes:params length:sizeof(params) atIndex:6];
        [enc setBuffer:ctx->ssm_prefill_buf offset:off_gate     * sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads,
                                              (NSUInteger)n_tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        metal_encode_matmul(ctx, enc, wo,
                             ctx->ssm_prefill_buf,
                             off_attn_out * sizeof(float),
                             ctx->ssm_prefill_buf,
                             off_wo_out * sizeof(float),
                             wo_type, wo_rows, wo_cols, n_tokens);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, scratch_base + off_wo_out, wo_floats * sizeof(float));
    return 0;
}

static int metal_dense_ffn_batch_stub(void *vctx, float *out,
                                      void *gate_buf, void *up_buf, void *down_buf,
                                      const float *X, int n_tokens,
                                      int dim, int hidden_dim,
                                      int gate_type, int up_type,
                                      int down_type, int act_type) {
    (void)vctx; (void)out; (void)gate_buf; (void)up_buf; (void)down_buf;
    (void)X; (void)n_tokens; (void)dim; (void)hidden_dim;
    (void)gate_type; (void)up_type; (void)down_type; (void)act_type;
    return -1;
}


static int metal_dense_ffn_batch_norm_resid(
        void *vctx, float *out,
        void *gate_buf, void *up_buf,
        void *down_buf, void *norm_buf,
        const float *X, int n_tokens,
        int dim, int hidden_dim,
        int gate_type, int up_type,
        int down_type, int act_type,
        float norm_eps)
{
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !out || !X || !gate_buf || !down_buf || !norm_buf)
        return -1;
    if (n_tokens <= 1 || dim <= 0 || hidden_dim <= 0) return -1;
    if (act_type != 0) return -1;
    if (gate_type < 0 || gate_type >= BN_METAL_MAX_TYPES) return -1;
    if (down_type < 0 || down_type >= BN_METAL_MAX_TYPES) return -1;
    if (!ctx->pipelines[gate_type] || !ctx->pipelines[down_type])
        return -1;
    if (!ctx->ssm_prefill_rmsnorm_pipeline ||
        !ctx->fwd_pipelines[BN_GPU_SHADER_SILU_GATE] ||
        !ctx->fwd_pipelines[BN_GPU_SHADER_RESIDUAL_ADD])
        return -1;

    int gateup_stacked = (up_buf == NULL);
    if (gateup_stacked && !ctx->ssm_prefill_silu_gate_stacked_pipeline)
        return -1;
    if (!gateup_stacked) {
        if (up_type < 0 || up_type >= BN_METAL_MAX_TYPES) return -1;
        if (!ctx->pipelines[up_type]) return -1;
    }

    BnMetalBuf *gate_w = (BnMetalBuf *)gate_buf;
    BnMetalBuf *up_w   = (BnMetalBuf *)up_buf;
    BnMetalBuf *down_w = (BnMetalBuf *)down_buf;
    BnMetalBuf *norm_w = (BnMetalBuf *)norm_buf;
    if (!gate_w->buf || !down_w->buf || !norm_w->buf) return -1;
    if (!gateup_stacked && (!up_w || !up_w->buf)) return -1;
    if (gate_w->q4_prepared || down_w->q4_prepared) return -1;
    if (!gateup_stacked && up_w->q4_prepared) return -1;
    if (gate_w->bias_offset || down_w->bias_offset) return -1;
    if (!gateup_stacked && up_w->bias_offset) return -1;

    size_t x_floats     = (size_t)n_tokens * (size_t)dim;
    size_t hidden_floats = (size_t)n_tokens * (size_t)hidden_dim;
    size_t gate_floats = gateup_stacked ? hidden_floats * 2 : hidden_floats;
    size_t up_floats   = gateup_stacked ? 0 : hidden_floats;
    size_t post_floats = gateup_stacked ? hidden_floats : 0;
    size_t total = x_floats * 3 + gate_floats + up_floats + post_floats;
    if (metal_ensure_ssm_prefill_buf(ctx, total * sizeof(float)) != 0)
        return -1;
    size_t off_x    = 0;
    size_t off_norm = off_x    + x_floats;
    size_t off_gate = off_norm + x_floats;
    size_t off_up   = off_gate + gate_floats;
    size_t off_post = off_up   + up_floats;
    size_t off_down = off_post + post_floats;

    float *scratch_base = (float *)[ctx->ssm_prefill_buf contents];
    memcpy(scratch_base + off_x, X, x_floats * sizeof(float));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        {
            uint32_t params[8] = { (uint32_t)dim, 0, 0, 0, 0, 0, 0, 0 };
            memcpy(&params[1], &norm_eps, sizeof(float));
            [enc setComputePipelineState:ctx->ssm_prefill_rmsnorm_pipeline];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_x    * sizeof(float) atIndex:0];
            [enc setBuffer:norm_w->buf offset:norm_w->offset atIndex:1];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_norm * sizeof(float) atIndex:2];
            [enc setBytes:params length:sizeof(params) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tokens, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        size_t off_ffn_down_in;
        if (gateup_stacked) {
            metal_encode_matmul(ctx, enc, gate_w,
                                 ctx->ssm_prefill_buf, off_norm * sizeof(float),
                                 ctx->ssm_prefill_buf, off_gate * sizeof(float),
                                 gate_type, hidden_dim * 2, dim, n_tokens);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            uint32_t params[8] = { (uint32_t)hidden_dim, (uint32_t)n_tokens,
                                   0, 0, 0, 0, 0, 0 };
            uint32_t wg_x = ((uint32_t)hidden_dim + 255u) / 256u;
            [enc setComputePipelineState:ctx->ssm_prefill_silu_gate_stacked_pipeline];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_gate * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_post * sizeof(float) atIndex:1];
            [enc setBytes:params length:sizeof(params) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, (NSUInteger)n_tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            off_ffn_down_in = off_post;
        } else {
            metal_encode_matmul(ctx, enc, gate_w,
                                 ctx->ssm_prefill_buf, off_norm * sizeof(float),
                                 ctx->ssm_prefill_buf, off_gate * sizeof(float),
                                 gate_type, hidden_dim, dim, n_tokens);
            metal_encode_matmul(ctx, enc, up_w,
                                 ctx->ssm_prefill_buf, off_norm * sizeof(float),
                                 ctx->ssm_prefill_buf, off_up * sizeof(float),
                                 up_type, hidden_dim, dim, n_tokens);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            uint32_t total_e = (uint32_t)hidden_floats;
            uint32_t params[8] = { total_e, 0, 0, 0, 0, 0, 0, 0 };
            uint32_t wg_x = (total_e + 255u) / 256u;
            [enc setComputePipelineState:ctx->fwd_pipelines[BN_GPU_SHADER_SILU_GATE]];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_gate * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_up   * sizeof(float) atIndex:1];
            [enc setBytes:params length:sizeof(params) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            off_ffn_down_in = off_gate;
        }

        metal_encode_matmul(ctx, enc, down_w,
                             ctx->ssm_prefill_buf, off_ffn_down_in * sizeof(float),
                             ctx->ssm_prefill_buf, off_down * sizeof(float),
                             down_type, dim, hidden_dim, n_tokens);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        {
            uint32_t total_e = (uint32_t)x_floats;
            uint32_t params[8] = { total_e, 0, 0, 0, 0, 0, 0, 0 };
            uint32_t wg_x = (total_e + 255u) / 256u;
            [enc setComputePipelineState:ctx->fwd_pipelines[BN_GPU_SHADER_RESIDUAL_ADD]];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_down * sizeof(float) atIndex:0];
            [enc setBuffer:ctx->ssm_prefill_buf offset:off_x    * sizeof(float) atIndex:1];
            [enc setBytes:params length:sizeof(params) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(wg_x, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    memcpy(out, scratch_base + off_down, x_floats * sizeof(float));
    return 0;
}

/* ── Vtable: execute (forward-pass) ────────────────────────────────── */

static int metal_execute(void *vctx, const void *ops_raw, int n_ops,
                         int readback_buf, float *out_host, int out_len)
{
    const BnGPUOp *ops = (const BnGPUOp *)ops_raw;
    BnMetalCtx *ctx = (BnMetalCtx *)vctx;
    if (!ctx || !ops || n_ops <= 0) return -1;
    if (ctx->gpu_profile < 0) {
        const char *env = getenv("BN_GPU_PROFILE");
        ctx->gpu_profile = env ? atoi(env) : 0;
    }

    double t0 = bn_platform_time_ms();
    double t_encode = 0, t_gpu = 0;
    int n_barriers = 0;
    BnMetalProfileShape matvec_shapes[32], q8_shapes[16], q4_shapes[16];
    int n_matvec_shapes = 0, n_q8_shapes = 0, n_q4_shapes = 0;
    BnMetalProfileShape timed_shapes[64];
    int n_timed_shapes = 0;
    int profile_each_op = ctx->gpu_profile >= 4;
    double shader_gpu_ms[BN_GPU_SHADER_COUNT];
    double shader_wall_ms[BN_GPU_SHADER_COUNT];
    int shader_profile_counts[BN_GPU_SHADER_COUNT];
    memset(matvec_shapes, 0, sizeof(matvec_shapes));
    memset(q8_shapes, 0, sizeof(q8_shapes));
    memset(q4_shapes, 0, sizeof(q4_shapes));
    memset(timed_shapes, 0, sizeof(timed_shapes));
    memset(shader_gpu_ms, 0, sizeof(shader_gpu_ms));
    memset(shader_wall_ms, 0, sizeof(shader_wall_ms));
    memset(shader_profile_counts, 0, sizeof(shader_profile_counts));
    ctx->q8_quant_dispatches = 0;
    ctx->q8k_quant_dispatches = 0;
    ctx->q8_matvec_dispatches = 0;
    ctx->q8_split_dispatches = 0;
    ctx->q8_gateup_dispatches = 0;
    int full_barriers = getenv("BN_METAL_FULL_BARRIERS") != NULL;
    int enable_barriers = getenv("BN_METAL_ENABLE_BARRIERS") != NULL ||
                          full_barriers;
    int disable_barriers = getenv("BN_METAL_DISABLE_BARRIERS") != NULL ||
                           !enable_barriers;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;

        /* Dependency tracking: only insert barriers on actual RAW/WAR/WAW conflicts.
         * Same logic as wgpu execute — track read/write buffer masks since last barrier. */
        uint32_t since_barrier_writes = 0;
        id<MTLComputePipelineState> current_pso = nil;

        for (int i = 0; i < n_ops; i++) {
            const BnGPUOp *op = &ops[i];
            int shader = bn_gpu_shader_from_op_code(op->op_code);

            /* COPY as compute shader — stays in compute encoder, no blit transitions */

            /* Determine pipeline */
            id<MTLComputePipelineState> pipeline = nil;
            if (shader == BN_GPU_SHADER_MATVEC) {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (ctx->q4_q8_enabled && op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf &&
                    wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_prepared_q8_matvec_pipeline) {
                    pipeline = ctx->q4_prepared_q8_matvec_pipeline;
                } else if (ctx->q4_q8_enabled && op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf &&
                    !wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline && ctx->q4_q8_matvec_pipeline) {
                    pipeline = ctx->q4_q8_matvec_pipeline;
                } else if (op->type >= 0 && op->type < BN_METAL_MAX_TYPES) {
                    pipeline = ctx->pipelines[op->type];
                }
            } else if (shader == BN_GPU_SHADER_RMSNORM &&
                       ctx->cpu_order_rmsnorm_enabled &&
                       ctx->cpu_order_rmsnorm_pipeline) {
                pipeline = ctx->cpu_order_rmsnorm_pipeline;
            } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                       ctx->q4_q8_enabled &&
                       op->p[6] &&
                       op->type == BN_GGUF_TENSOR_Q4_0 &&
                       op->W_buf &&
                       ctx->q8_quant_pipeline &&
                       ((BnMetalBuf *)op->W_buf)->q4_prepared &&
                       ctx->q4_prepared_q8_gateup_pipeline) {
                pipeline = ctx->q4_prepared_q8_gateup_pipeline;
            } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                       ctx->q4_q8_enabled &&
                       op->p[6] &&
                       op->type == BN_GGUF_TENSOR_Q4_0 &&
                       op->W_buf &&
                       !((BnMetalBuf *)op->W_buf)->q4_prepared &&
                       ctx->q8_quant_pipeline &&
                       ctx->q4_q8_gateup_pipeline) {
                pipeline = ctx->q4_q8_gateup_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       ctx->q4_q8_enabled &&
                       (op->flags & 1u) &&
                       op->type == BN_GGUF_TENSOR_Q4_0 &&
                       op->W_buf &&
                       ctx->q8_quant_pipeline &&
                       ((BnMetalBuf *)op->W_buf)->q4_prepared &&
                       ctx->q4_prepared_q8_split_pipeline) {
                pipeline = ctx->q4_prepared_q8_split_pipeline;
            } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                       ctx->q4_q8_enabled &&
                       (op->flags & 1u) &&
                       op->type == BN_GGUF_TENSOR_Q4_0 &&
                       op->W_buf &&
                       !((BnMetalBuf *)op->W_buf)->q4_prepared &&
                       ctx->q8_quant_pipeline &&
                       ctx->q4_q8_split_pipeline) {
                pipeline = ctx->q4_q8_split_pipeline;
            } else if (shader > 0 && shader < BN_GPU_SHADER_COUNT) {
                pipeline = ctx->fwd_pipelines[shader];
            }
            if (!pipeline) continue;

            if (ctx->gpu_profile >= 2 &&
                (shader == BN_GPU_SHADER_MATVEC ||
                 shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                 shader == BN_GPU_SHADER_FUSED_GATEUP_SILU)) {
                uint32_t n_tokens = shader == BN_GPU_SHADER_MATVEC && op->p[2]
                    ? op->p[2]
                    : 1;
                uint32_t rows = shader == BN_GPU_SHADER_MATVEC_SPLIT
                    ? op->p[0]
                    : (uint32_t)op->rows;
                if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU && op->p[0])
                    rows = op->p[0];
                metal_profile_add_shape(matvec_shapes, &n_matvec_shapes, 32,
                                        shader, rows, (uint32_t)op->cols,
                                        n_tokens);
            }

            /* Compute this op's read/write buffer masks. */
            uint32_t op_reads = 0, op_writes = 0;
            if (bn_gpu_shader_access_masks(op, shader, &op_reads,
                                           &op_writes) != 0)
                continue;

            /* Insert barrier only on RAW conflict (read-after-write).
             * WAR and WAW don't need barriers — Metal dispatches execute in
             * submission order within a compute command encoder, so reads
             * always complete before subsequent writes to the same buffer. */
            int conflict = disable_barriers ? 0
                : (full_barriers ? (since_barrier_writes != 0)
                                 : (op_reads & since_barrier_writes));
            if (conflict && enc) {
                /* Use resource-specific barriers for less stalling */
                /* Collect MTLBuffer pointers for written buffers that this op reads */
                id<MTLBuffer> barrier_bufs[BN_GPU_BUF_COUNT];
                int n_bbuf = 0;
                for (int b = 0; b < BN_GPU_BUF_COUNT && b < 23; b++) {
                    if ((since_barrier_writes & (1u << b)) &&
                        ((op_reads | op_writes) & (1u << b)) &&
                        ctx->act_bufs[b]) {
                        barrier_bufs[n_bbuf++] = ctx->act_bufs[b];
                    }
                }
                if (n_bbuf > 0)
                    [enc memoryBarrierWithResources:barrier_bufs count:(NSUInteger)n_bbuf];
                else
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                since_barrier_writes = 0;
                n_barriers++;
            }

            since_barrier_writes |= op_writes;

            /* Start compute encoder if needed */
            if (!cmd)
                cmd = [ctx->queue commandBuffer];
            if (!enc) {
                enc = [cmd computeCommandEncoder];
                current_pso = nil;
            }
            BnMetalBuf *pre_wbuf = (BnMetalBuf *)op->W_buf;
            int q4_q8_deferred_pso =
                ctx->q4_q8_enabled &&
                op->type == BN_GGUF_TENSOR_Q4_0 &&
                ctx->q8_quant_pipeline &&
                ((shader == BN_GPU_SHADER_MATVEC && op->p[6] &&
                  pre_wbuf &&
                  ((pre_wbuf->q4_prepared &&
                    ctx->q4_prepared_q8_matvec_pipeline) ||
                   (!pre_wbuf->q4_prepared &&
                    ctx->q4_q8_matvec_pipeline))) ||
                 (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                  (op->flags & 1u) &&
                  pre_wbuf &&
                  ((pre_wbuf->q4_prepared &&
                    ctx->q4_prepared_q8_split_pipeline) ||
                   (!pre_wbuf->q4_prepared &&
                    ctx->q4_q8_split_pipeline))) ||
                 (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                  op->p[6] &&
                  pre_wbuf &&
                  ((pre_wbuf->q4_prepared &&
                    ctx->q4_prepared_q8_gateup_pipeline) ||
                   (!pre_wbuf->q4_prepared &&
                    ctx->q4_q8_gateup_pipeline))));
            /* Skip redundant PSO switch — avoids GPU instruction cache flush */
            if (!q4_q8_deferred_pso && pipeline != current_pso) {
                [enc setComputePipelineState:pipeline];
                current_pso = pipeline;
            }

            /* Set buffers per shader type + setBytes for uniforms */
            uint32_t params[BN_GPU_OP_PARAMS];
            memcpy(params, op->p, sizeof(params));

            /* Inject fused bias for matvec */
            if (shader == BN_GPU_SHADER_MATVEC && op->W_buf) {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
            }

            switch (shader) {
            case BN_GPU_SHADER_MATVEC: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                if (ctx->q4_q8_enabled && op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_prepared_q8_matvec_pipeline) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_q8_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        return -1;
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(q8_shapes, &n_q8_shapes, 16,
                                                shader, 0, (uint32_t)op->cols,
                                                n_tokens);
                        metal_profile_add_shape(q4_shapes, &n_q4_shapes, 16,
                                                shader, (uint32_t)op->rows,
                                                (uint32_t)op->cols, n_tokens);
                    }
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, n_tokens);
                    ctx->q8_matvec_dispatches++;
                    [enc setComputePipelineState:ctx->q4_prepared_q8_matvec_pipeline];
                    current_pso = ctx->q4_prepared_q8_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (ctx->q4_q8_enabled && op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    !wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline && ctx->q4_q8_matvec_pipeline) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_q8_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        return -1;
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(q8_shapes, &n_q8_shapes, 16,
                                                shader, 0, (uint32_t)op->cols,
                                                n_tokens);
                        metal_profile_add_shape(q4_shapes, &n_q4_shapes, 16,
                                                shader, (uint32_t)op->rows,
                                                (uint32_t)op->cols, n_tokens);
                    }
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, n_tokens);
                    ctx->q8_matvec_dispatches++;
                    [enc setComputePipelineState:ctx->q4_q8_matvec_pipeline];
                    current_pso = ctx->q4_q8_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (op->type == BN_GGUF_TENSOR_Q6_K &&
                           getenv("BN_METAL_ENABLE_Q6_Q8K") &&
                           ctx->q8k_quant_pipeline &&
                           ctx->q6_q8k_matvec_pipeline &&
                           (op->cols % 256) == 0) {
                    uint32_t n_tokens = params[2] ? params[2] : 1;
                    if (ensure_q8k_scratch(ctx, op->cols, (int)n_tokens) != 0)
                        return -1;
                    metal_encode_q8k_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                           (uint32_t)op->cols, n_tokens);
                    [enc setComputePipelineState:ctx->q6_q8k_matvec_pipeline];
                    current_pso = ctx->q6_q8k_matvec_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->q8_bsums_buf offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:4];
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                } else {
                    if (wbuf->q4_prepared)
                        return -1;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                }
                break;
            }
            case BN_GPU_SHADER_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (ctx->cpu_order_rmsnorm_enabled &&
                    ctx->cpu_order_rmsnorm_pipeline) {
                    [enc setComputePipelineState:ctx->cpu_order_rmsnorm_pipeline];
                    current_pso = ctx->cpu_order_rmsnorm_pipeline;
                }
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                if (wbuf)
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:1];
                else
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_ROPE: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_GQA_SCORES: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_KEY_CACHE] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ATT] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_SOFTMAX: {
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ATT] offset:0 atIndex:0];
                [enc setBytes:params length:sizeof(params) atIndex:1];
                break;
            }
            case BN_GPU_SHADER_GQA_COMBINE: {
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ATT] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_VALUE_CACHE] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_SILU_GATE:
            case BN_GPU_SHADER_RELU2_GATE: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_RESIDUAL_ADD: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_BIAS_ADD: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_RESIDUAL_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_WEIGHTED_ADD: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_SSM_CONV_SILU: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_CONV_STATE] offset:0 atIndex:1];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_SSM_L2NORM: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_SSM_ALPHA_BETA: {
                BnMetalBuf *dt_buf = (BnMetalBuf *)op->W_buf;
                if (!dt_buf) continue;
                void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
                BnMetalBuf *a_wbuf = (BnMetalBuf *)a_ptr;
                if (!a_wbuf) continue;
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_BETA] offset:0 atIndex:1];
                [enc setBuffer:dt_buf->buf offset:dt_buf->offset atIndex:2];
                [enc setBuffer:a_wbuf->buf offset:a_wbuf->offset atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT: {
                BnMetalBuf *dt_buf = (BnMetalBuf *)op->W_buf;
                if (!dt_buf) continue;
                void *a_ptr = (void *)(uintptr_t)((uint64_t)op->p[6] | ((uint64_t)op->p[7] << 32));
                BnMetalBuf *a_wbuf = (BnMetalBuf *)a_ptr;
                if (!a_wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_BETA] offset:0 atIndex:2];
                [enc setBuffer:dt_buf->buf offset:dt_buf->offset atIndex:3];
                [enc setBuffer:a_wbuf->buf offset:a_wbuf->offset atIndex:4];
                [enc setBytes:params length:sizeof(params) atIndex:5];
                break;
            }
            case BN_GPU_SHADER_SSM_DELTA: {
                int v_buf = op->p[7] ? op->buf_in : BN_GPU_BUF_SSM_V;
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_STATE] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                [enc setBuffer:ctx->act_bufs[v_buf] offset:0 atIndex:4];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_ALPHA] offset:0 atIndex:5];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_SSM_BETA] offset:0 atIndex:6];
                [enc setBytes:params length:sizeof(params) atIndex:7];
                break;
            }
            case BN_GPU_SHADER_SSM_GATE: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_PER_HEAD_RMSNORM: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_DEINTERLEAVE_Q: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_SIGMOID_GATE: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_FLASH_ATTN: {
                /* Fused: Q(buf_in) + key_cache + value_cache → xb(buf_out) */
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_KEY_CACHE] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_VALUE_CACHE] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_COPY: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:1];
                [enc setBytes:params length:sizeof(params) atIndex:2];
                break;
            }
            case BN_GPU_SHADER_MATVEC_SPLIT: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
                if (ctx->q4_q8_enabled &&
                    (op->flags & 1u) &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_prepared_q8_split_pipeline) {
                    if (ensure_q8_scratch(ctx, op->cols, 1) != 0)
                        return -1;
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->q8_split_dispatches++;
                    [enc setComputePipelineState:ctx->q4_prepared_q8_split_pipeline];
                    current_pso = ctx->q4_prepared_q8_split_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[op->rows] offset:0 atIndex:5];
                    [enc setBytes:params length:sizeof(params) atIndex:6];
                } else if (ctx->q4_q8_enabled &&
                    (op->flags & 1u) &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    !wbuf->q4_prepared &&
                       ctx->q8_quant_pipeline &&
                       ctx->q4_q8_split_pipeline) {
                    if (ensure_q8_scratch(ctx, op->cols, 1) != 0)
                        return -1;
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(q8_shapes, &n_q8_shapes, 16,
                                                shader, 0, (uint32_t)op->cols,
                                                1);
                        metal_profile_add_shape(q4_shapes, &n_q4_shapes, 16,
                                                shader, op->p[0],
                                                (uint32_t)op->cols, 1);
                    }
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->q8_split_dispatches++;
                    [enc setComputePipelineState:ctx->q4_q8_split_pipeline];
                    current_pso = ctx->q4_q8_split_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:4];
                    [enc setBuffer:ctx->act_bufs[op->rows] offset:0 atIndex:5];
                    [enc setBytes:params length:sizeof(params) atIndex:6];
                } else {
                    if (wbuf->q4_prepared)
                        return -1;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];  // out0
                    [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];  // out1
                    [enc setBuffer:ctx->act_bufs[op->rows] offset:0 atIndex:4];     // out2
                    [enc setBytes:params length:sizeof(params) atIndex:5];
                }
                break;
            }
            case BN_GPU_SHADER_ROPE_QK: {
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:0];   // Q
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:1];  // K (KEY_CACHE)
                [enc setBuffer:ctx->act_bufs[BN_GPU_BUF_ROPE_FREQ] offset:0 atIndex:2];
                [enc setBytes:params length:sizeof(params) atIndex:3];
                break;
            }
            case BN_GPU_SHADER_FUSED_GATEUP_SILU: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                if (wbuf->bias_offset > 0) params[4] = wbuf->bias_offset;
                if (ctx->q4_q8_enabled &&
                    op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_prepared_q8_gateup_pipeline) {
                    if (ensure_q8_scratch(ctx, op->cols, 1) != 0)
                        return -1;
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->q8_gateup_dispatches++;
                    [enc setComputePipelineState:ctx->q4_prepared_q8_gateup_pipeline];
                    current_pso = ctx->q4_prepared_q8_gateup_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else if (ctx->q4_q8_enabled &&
                    op->p[6] &&
                    op->type == BN_GGUF_TENSOR_Q4_0 &&
                    !wbuf->q4_prepared &&
                    ctx->q8_quant_pipeline &&
                    ctx->q4_q8_gateup_pipeline) {
                    if (ensure_q8_scratch(ctx, op->cols, 1) != 0)
                        return -1;
                    if (ctx->gpu_profile >= 2) {
                        metal_profile_add_shape(q8_shapes, &n_q8_shapes, 16,
                                                shader, 0, (uint32_t)op->cols,
                                                1);
                        metal_profile_add_shape(q4_shapes, &n_q4_shapes, 16,
                                                shader, op->p[2],
                                                (uint32_t)op->cols, 1);
                    }
                    metal_encode_q8_quant(enc, ctx, ctx->act_bufs[op->buf_in],
                                          (uint32_t)op->cols, 1);
                    ctx->q8_gateup_dispatches++;
                    [enc setComputePipelineState:ctx->q4_q8_gateup_pipeline];
                    current_pso = ctx->q4_q8_gateup_pipeline;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->q8_buf offset:0 atIndex:1];
                    [enc setBuffer:ctx->q8_scales_buf offset:0 atIndex:2];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:3];
                    [enc setBytes:params length:sizeof(params) atIndex:4];
                } else {
                    if (wbuf->q4_prepared)
                        return -1;
                    [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                    [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                    [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                    [enc setBytes:params length:sizeof(params) atIndex:3];
                }
                break;
            }
            case BN_GPU_SHADER_Q4K_MATVEC_SPLIT: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                [enc setBytes:params length:sizeof(params) atIndex:4];
                break;
            }
            case BN_GPU_SHADER_Q5K_MATVEC_SPLIT: {
                BnMetalBuf *wbuf = (BnMetalBuf *)op->W_buf;
                if (!wbuf) continue;
                int out2_slot = (op->p[3] > 0u) ? op->rows : op->buf_aux;
                [enc setBuffer:wbuf->buf offset:wbuf->offset atIndex:0];
                [enc setBuffer:ctx->act_bufs[op->buf_in] offset:0 atIndex:1];
                [enc setBuffer:ctx->act_bufs[op->buf_out] offset:0 atIndex:2];
                [enc setBuffer:ctx->act_bufs[op->buf_aux] offset:0 atIndex:3];
                [enc setBuffer:ctx->act_bufs[out2_slot] offset:0 atIndex:4];
                [enc setBytes:params length:sizeof(params) atIndex:5];
                break;
            }
            default: continue;
            }

            /* Compute workgroup count (same logic as wgpu) */
            uint32_t wg_x = 1, wg_y = 1;
            uint32_t tile_rows = 32;
            uint32_t threads_per_tg = 256;
            BnMetalBuf *grid_wbuf = (BnMetalBuf *)op->W_buf;
            int q4_q8_tile =
                grid_wbuf &&
                ctx->q4_q8_enabled &&
                op->type == BN_GGUF_TENSOR_Q4_0 &&
                !grid_wbuf->q4_prepared &&
                ((shader == BN_GPU_SHADER_MATVEC &&
                  op->p[6] && ctx->q8_quant_pipeline &&
                  ctx->q4_q8_matvec_pipeline) ||
                 (shader == BN_GPU_SHADER_MATVEC_SPLIT &&
                  (op->flags & 1u) && ctx->q8_quant_pipeline &&
                  ctx->q4_q8_split_pipeline) ||
                 (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU &&
                  op->p[6] && ctx->q8_quant_pipeline &&
                  ctx->q4_q8_gateup_pipeline));
            if (q4_q8_tile) {
                tile_rows = 16;
                threads_per_tg = 128;
            }
            switch (shader) {
            case BN_GPU_SHADER_MATVEC: {
                if (op->p[3] > 0) {
                    uint32_t tiled_rows = ((uint32_t)op->rows + tile_rows - 1) / tile_rows;
                    wg_x = op->p[3];
                    wg_y = (tiled_rows + op->p[3] - 1) / op->p[3];
                } else {
                    wg_x = ((uint32_t)op->rows + tile_rows - 1) / tile_rows;
                    wg_y = op->p[2];
                    if (wg_y == 0) wg_y = 1;
                }
                break;
            }
            case BN_GPU_SHADER_RMSNORM:
            case BN_GPU_SHADER_RESIDUAL_RMSNORM:
            case BN_GPU_SHADER_SSM_ALPHA_BETA:
            case BN_GPU_SHADER_SSM_ALPHA_BETA_SPLIT:
                wg_x = 1;
                break;
            case BN_GPU_SHADER_ROPE:
            case BN_GPU_SHADER_SOFTMAX:
            case BN_GPU_SHADER_GQA_COMBINE:
                wg_x = op->p[0];
                break;
            case BN_GPU_SHADER_GQA_SCORES:
                wg_x = op->p[0];
                wg_y = (op->p[2] + 7) / 8;
                break;
            case BN_GPU_SHADER_SILU_GATE:
            case BN_GPU_SHADER_RELU2_GATE:
            case BN_GPU_SHADER_RESIDUAL_ADD:
            case BN_GPU_SHADER_BIAS_ADD:
            case BN_GPU_SHADER_WEIGHTED_ADD:
            case BN_GPU_SHADER_SSM_CONV_SILU:
                wg_x = (op->p[0] + 255) / 256;
                break;
            case BN_GPU_SHADER_SSM_L2NORM:
                wg_x = (uint32_t)op->rows;
                break;
            case BN_GPU_SHADER_SSM_DELTA:
            case BN_GPU_SHADER_SSM_GATE:
                wg_x = (uint32_t)op->rows;
                break;
            case BN_GPU_SHADER_PER_HEAD_RMSNORM:
                wg_x = (uint32_t)op->rows;
                break;
            case BN_GPU_SHADER_DEINTERLEAVE_Q:
            case BN_GPU_SHADER_SIGMOID_GATE:
                wg_x = (op->p[0] + 255) / 256;
                break;
            case BN_GPU_SHADER_FLASH_ATTN:
                wg_x = op->p[0];  /* one head per threadgroup */
                break;
            case BN_GPU_SHADER_COPY:
                wg_x = (op->p[2] + 255) / 256;
                break;
            case BN_GPU_SHADER_MATVEC_SPLIT:
                wg_x = (op->p[0] + tile_rows - 1) / tile_rows;
                break;
            case BN_GPU_SHADER_ROPE_QK:
                wg_x = op->p[0] + op->p[4];   // n_q_heads + n_kv_heads
                break;
            case BN_GPU_SHADER_FUSED_GATEUP_SILU:
                wg_x = (op->p[2] + tile_rows - 1) / tile_rows;
                break;
            case BN_GPU_SHADER_Q4K_MATVEC_SPLIT:
            case BN_GPU_SHADER_Q5K_MATVEC_SPLIT:
                wg_x = (op->p[0] + 31) / 32;  // total_rows / 32
                break;
            }

            if (wg_x == 0) wg_x = 1;
            MTLSize tpg = MTLSizeMake(threads_per_tg, 1, 1);
            MTLSize grid = MTLSizeMake(wg_x, wg_y, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];

            if (profile_each_op && shader >= 0 && shader < BN_GPU_SHADER_COUNT) {
                double op_wall0 = bn_platform_time_ms();
                [enc endEncoding];
                enc = nil;
                id<MTLCommandBuffer> done_cmd = cmd;
                [done_cmd commit];
                [done_cmd waitUntilCompleted];
                double op_wall1 = bn_platform_time_ms();
                double gpu_ms = 0.0;
                double gpu_start = done_cmd.GPUStartTime;
                double gpu_end = done_cmd.GPUEndTime;
                if (gpu_end > gpu_start)
                    gpu_ms = (gpu_end - gpu_start) * 1000.0;
                shader_gpu_ms[shader] += gpu_ms;
                shader_wall_ms[shader] += op_wall1 - op_wall0;
                shader_profile_counts[shader]++;
                if (shader == BN_GPU_SHADER_MATVEC ||
                    shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                    shader == BN_GPU_SHADER_FUSED_GATEUP_SILU ||
                    shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT) {
                    uint32_t rows = (uint32_t)op->rows;
                    uint32_t cols = (uint32_t)op->cols;
                    uint32_t aux = 1;
                    if (shader == BN_GPU_SHADER_MATVEC) {
                        aux = op->p[2] ? op->p[2] : 1;
                    } else if (shader == BN_GPU_SHADER_MATVEC_SPLIT ||
                               shader == BN_GPU_SHADER_Q4K_MATVEC_SPLIT) {
                        rows = op->p[0];
                    } else if (shader == BN_GPU_SHADER_FUSED_GATEUP_SILU) {
                        rows = op->p[2];
                    }
                    metal_profile_add_shape_time(
                        timed_shapes, &n_timed_shapes, 64, shader, rows,
                        cols, aux, gpu_ms, op_wall1 - op_wall0);
                }
                cmd = nil;
                current_pso = nil;
            }
        }

        if (enc) [enc endEncoding];

        t_encode = bn_platform_time_ms();

        if (cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        t_gpu = bn_platform_time_ms();
    }

    /* Readback: unified memory — just memcpy */
    if (out_host && out_len > 0 && readback_buf >= 0
        && readback_buf < BN_GPU_BUF_COUNT && ctx->act_bufs[readback_buf]) {
        size_t readback_size = (size_t)out_len * sizeof(float);
        memcpy(out_host, [ctx->act_bufs[readback_buf] contents], readback_size);
    }

    double t1 = bn_platform_time_ms();

    /* GPU profiling */
    if (ctx->gpu_profile >= 1 && (ctx->gpu_frame < 5 || (ctx->gpu_frame % 50 == 0))) {
        fprintf(stderr, "[gpu:metal:profile] frame=%d ops=%d q8=%d q8m=%d q8s=%d q8g=%d q8k=%d barriers=%d encode=%.1fms gpu=%.1fms readback=%.1fms total=%.1fms\n",
                ctx->gpu_frame, n_ops, ctx->q8_quant_dispatches,
                ctx->q8_matvec_dispatches, ctx->q8_split_dispatches,
                ctx->q8_gateup_dispatches, ctx->q8k_quant_dispatches,
                n_barriers,
                t_encode - t0, t_gpu - t_encode, t1 - t_gpu, t1 - t0);
    }
    /* Per-op-type breakdown (BN_GPU_PROFILE>=2, frame 1 only) */
    if (ctx->gpu_profile >= 2 && ctx->gpu_frame == 1) {
        int cat_count[BN_GPU_SHADER_COUNT]; memset(cat_count, 0, sizeof(cat_count));
        for (int i = 0; i < n_ops; i++) {
            int s = bn_gpu_shader_from_op_code(ops[i].op_code);
            if (s >= 0 && s < BN_GPU_SHADER_COUNT) cat_count[s]++;
        }
        fprintf(stderr, "[gpu:metal:breakdown] --- op counts ---\n");
        for (int s = 0; s < BN_GPU_SHADER_COUNT; s++) {
            if (cat_count[s] > 0)
                fprintf(stderr, "  %-16s: %3d ops\n",
                        metal_shader_profile_name(s), cat_count[s]);
        }
        if (n_q8_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- q8 activation quant shapes ---\n");
            for (int i = 0; i < n_q8_shapes; i++) {
                fprintf(stderr, "  %-16s: %3d dispatches cols=%u tokens=%u\n",
                        metal_shader_profile_name(q8_shapes[i].shader),
                        q8_shapes[i].count, q8_shapes[i].cols,
                        q8_shapes[i].aux);
            }
        }
        if (n_q4_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- q4/q8 matvec shapes ---\n");
            for (int i = 0; i < n_q4_shapes; i++) {
                fprintf(stderr, "  %-16s: %3d dispatches rows=%u cols=%u tokens=%u\n",
                        metal_shader_profile_name(q4_shapes[i].shader),
                        q4_shapes[i].count, q4_shapes[i].rows,
                        q4_shapes[i].cols, q4_shapes[i].aux);
            }
        }
        if (n_matvec_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- matvec shapes ---\n");
            for (int i = 0; i < n_matvec_shapes; i++) {
                fprintf(stderr, "  %-16s: %3d ops rows=%u cols=%u tokens=%u\n",
                        metal_shader_profile_name(matvec_shapes[i].shader),
                        matvec_shapes[i].count, matvec_shapes[i].rows,
                        matvec_shapes[i].cols, matvec_shapes[i].aux);
            }
        }
    }
    /* Per-op command-buffer timing (BN_GPU_PROFILE>=4, frame 1 only).
     * This intentionally changes submission granularity and is diagnostic-only. */
    if (ctx->gpu_profile >= 4 && ctx->gpu_frame == 1) {
        fprintf(stderr, "[gpu:metal:breakdown] --- per-op shader timing ---\n");
        for (int s = 0; s < BN_GPU_SHADER_COUNT; s++) {
            if (shader_profile_counts[s] > 0) {
                double shown_gpu = shader_gpu_ms[s] > 0.0
                    ? shader_gpu_ms[s]
                    : shader_wall_ms[s];
                fprintf(stderr, "  %-16s: %3d ops gpu=%.3fms wall=%.3fms avg=%.3fms\n",
                        metal_shader_profile_name(s),
                        shader_profile_counts[s],
                        shown_gpu,
                        shader_wall_ms[s],
                        shown_gpu / (double)shader_profile_counts[s]);
            }
        }
        if (n_timed_shapes > 0) {
            fprintf(stderr, "[gpu:metal:breakdown] --- per-shape timing ---\n");
            for (int i = 0; i < n_timed_shapes; i++) {
                double shown_gpu = timed_shapes[i].gpu_ms > 0.0
                    ? timed_shapes[i].gpu_ms
                    : timed_shapes[i].wall_ms;
                fprintf(stderr, "  %-16s: %3d ops rows=%u cols=%u tokens=%u gpu=%.3fms avg=%.3fms\n",
                        metal_shader_profile_name(timed_shapes[i].shader),
                        timed_shapes[i].count,
                        timed_shapes[i].rows,
                        timed_shapes[i].cols,
                        timed_shapes[i].aux,
                        shown_gpu,
                        shown_gpu / (double)timed_shapes[i].count);
            }
        }
    }
    ctx->gpu_frame++;

    return 0;
}

/* ── Public API ────────────────────────────────────────────────────── */

BnGPUBackend *bn_gpu_metal_create(const char *shader_dir)
{
    BnMetalCtx *ctx = (BnMetalCtx *)calloc(1, sizeof(BnMetalCtx));
    if (!ctx) return NULL;
    ctx->gpu_profile = -1;

    @autoreleasepool {
        /* Get default Metal device */
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "[bn:gpu:metal] no Metal device found\n");
            free(ctx);
            return NULL;
        }

        fprintf(stderr, "[bn:gpu:metal] device: %s\n",
                [[ctx->device name] UTF8String]);

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "[bn:gpu:metal] failed to create command queue\n");
            free(ctx);
            return NULL;
        }

        /* Store shader directory */
        const char *dir = shader_dir ? shader_dir : "shaders/metal/";
        snprintf(ctx->shader_dir, sizeof(ctx->shader_dir), "%s", dir);

        /* Compile matvec pipelines for all supported quant types */
        int compiled = 0;
        for (int i = 0; i < N_SUPPORTED_TYPES; i++) {
            int type = supported_types[i];
            if (compile_matvec_pipeline(ctx, type, dir) == 0)
                compiled++;
        }
        fprintf(stderr, "[bn:gpu:metal] compiled %d/%d matvec pipelines\n",
                compiled, N_SUPPORTED_TYPES);

        if (!getenv("BN_GPU_Q4_Q8") &&
            !getenv("BN_METAL_DISABLE_Q4_Q8_DEFAULT")) {
            setenv("BN_GPU_Q4_Q8", "1", 1);
            if (!getenv("BN_GPU_Q4_Q8_FROM_LAYER"))
                setenv("BN_GPU_Q4_Q8_FROM_LAYER", "0", 1);
        }
        ctx->q4_q8_enabled = getenv("BN_GPU_Q4_Q8") ? 1 : 0;
        ctx->q8_barriers_enabled = getenv("BN_METAL_Q8_BARRIERS") ? 1 : 0;
        ctx->cpu_order_rmsnorm_enabled =
            getenv("BN_METAL_CPU_ORDER_RMSNORM") ? 1 : 0;

        if (getenv("BN_METAL_ENABLE_Q6_Q8K")) {
            ctx->q8k_quant_pipeline = compile_shader(ctx, dir,
                "q8k_quantize.metal", "q8k_quantize");
            ctx->q6_q8k_matvec_pipeline = compile_shader(ctx, dir,
                "q6k_q8k_matvec.metal", "q6k_q8k_matvec");
            if (!ctx->q8k_quant_pipeline || !ctx->q6_q8k_matvec_pipeline) {
                fprintf(stderr, "[bn:gpu:metal] --metal-enable-q6-q8k requested "
                        "but required Q8_K/Q6_K pipelines failed to compile\n");
                free(ctx);
                return NULL;
            }
        }
        if (ctx->q4_q8_enabled) {
            ctx->q8_quant_pipeline = compile_shader(ctx, dir,
                "q8_quantize.metal", "q8_quantize");
            ctx->q4_q8_matvec_pipeline = compile_shader(ctx, dir,
                "q4_native_q8_prequant_matvec.metal",
                "q4_native_q8_prequant_matvec");
            ctx->q4_prepared_q8_matvec_pipeline = compile_shader(ctx, dir,
                "q4_prepared_q8_matvec.metal",
                "q4_prepared_q8_matvec");
            ctx->q4_prepared_q8_split_pipeline = compile_shader(ctx, dir,
                "q4_prepared_q8_split.metal",
                "q4_prepared_q8_split");
            ctx->q4_prepared_q8_gateup_pipeline = compile_shader(ctx, dir,
                "q4_prepared_q8_gateup.metal",
                "q4_prepared_q8_gateup");
            ctx->q4_q8_split_pipeline = compile_shader(ctx, dir,
                "q4_matvec_split_q8_prequant.metal",
                "q4_matvec_split_q8_prequant");
            ctx->q4_q8_gateup_pipeline = compile_shader(ctx, dir,
                "q4_fused_gateup_silu_q8_prequant.metal",
                "q4_fused_gateup_silu_q8_prequant");
        }

        /* Build vtable */
        BnGPUBackend *gpu = (BnGPUBackend *)calloc(1, sizeof(BnGPUBackend));
        if (!gpu) {
            free(ctx);
            return NULL;
        }
        gpu->buffer_create        = metal_buffer_create;
        gpu->buffer_create_biased = metal_buffer_create_biased;
        gpu->buffer_create_stacked2 = metal_buffer_create_stacked2;
        gpu->buffer_create_stacked3 = metal_buffer_create_stacked3;
        gpu->buffer_create_stacked3_biased = metal_buffer_create_stacked3_biased;
        gpu->buffer_destroy       = metal_buffer_destroy;
        gpu->matvec               = metal_matvec;
        gpu->matmul               = metal_matmul;
        gpu->matmul_batch         = metal_matmul_batch;
        gpu->matvec_batch         = metal_matvec_batch;
        gpu->dense_ffn            = metal_dense_ffn;
        gpu->kv_cache_init        = metal_kv_cache_init;
        gpu->kv_cache_write       = metal_kv_cache_write;
        gpu->prefill_kv_prep      = metal_prefill_kv_prep;
        gpu->prefill_begin_batch  = metal_prefill_begin_batch;
        gpu->prefill_flush        = metal_prefill_flush;
        gpu->prefill_ssm_layer    = metal_prefill_ssm_layer;
        gpu->prefill_attention    = metal_prefill_attention;
        gpu->prefill_attention_wo = metal_prefill_attention_wo;
        gpu->dense_ffn_batch         = metal_dense_ffn_batch_stub;
        gpu->dense_ffn_batch_norm_resid = metal_dense_ffn_batch_norm_resid;
        gpu->execute              = metal_execute;
        gpu->init_activations     = metal_init_activations;
        gpu->free_activations     = metal_free_activations;
        gpu->write_activation     = metal_write_activation;
        gpu->read_activation      = metal_read_activation;
        gpu->ctx                  = ctx;
        gpu->caps                 = BN_GPU_CAP_FLASH_ATTN |
                                    BN_GPU_CAP_Q4_MATVEC_SPLIT |
                                    BN_GPU_CAP_Q4K_MATVEC_SPLIT |
                                    BN_GPU_CAP_Q4_FUSED_GATEUP_SILU;
        gpu->kind                 = BN_GPU_BACKEND_METAL;

        return gpu;
    }
}

void bn_gpu_metal_destroy(BnGPUBackend *gpu)
{
    if (!gpu) return;

    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    if (ctx) {
        metal_free_activations(ctx);

        /* Release matvec pipelines */
        for (int i = 0; i < BN_METAL_MAX_TYPES; i++)
            ctx->pipelines[i] = nil;

        ctx->x_buf = nil;
        ctx->out_buf = nil;
        ctx->ssm_prefill_buf = nil;

        /* Release slab */
        ctx->slab_buf = nil;
        free(ctx->slab_free);

        ctx->queue = nil;
        ctx->device = nil;

        free(ctx);
    }
    free(gpu);
}

int bn_gpu_metal_init_slab(BnGPUBackend *gpu, size_t size_mb)
{
    if (!gpu || !gpu->ctx || size_mb == 0) return -1;
    return slab_init((BnMetalCtx *)gpu->ctx, size_mb * 1024 * 1024);
}

void bn_gpu_metal_set_mmap_range(BnGPUBackend *gpu, const void *base, size_t size)
{
    if (!gpu || !gpu->ctx) return;
    BnMetalCtx *ctx = (BnMetalCtx *)gpu->ctx;
    ctx->mmap_base = base;
    ctx->mmap_size = size;
}

#endif /* BN_ENABLE_METAL */
