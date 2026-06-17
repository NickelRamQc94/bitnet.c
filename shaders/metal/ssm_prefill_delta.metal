#include <metal_stdlib>
using namespace metal;

constant bool FC_FUSE_AB [[function_constant(0)]];
constant bool FC_HAVE_FUSE_AB = is_function_constant_defined(FC_FUSE_AB);
constant bool DO_FUSE_AB = FC_HAVE_FUSE_AB && FC_FUSE_AB;

constant bool FC_FUSE_L2NORM [[function_constant(1)]];
constant bool FC_HAVE_FUSE_L2NORM = is_function_constant_defined(FC_FUSE_L2NORM);
constant bool DO_FUSE_L2NORM = FC_HAVE_FUSE_L2NORM && FC_FUSE_L2NORM;

kernel void ssm_prefill_delta(device float       *state   [[buffer(0)]],
                              device float       *out     [[buffer(1)]],
                              device const float *qkv     [[buffer(2)]],
                              device const float *alpha   [[buffer(3)]],
                              device const float *beta    [[buffer(4)]],
                              constant uint      *p       [[buffer(5)]],
                              device const float *dt_bias [[buffer(6)]],
                              device const float *a_log   [[buffer(7)]],
                              uint3 wid [[threadgroup_position_in_grid]],
                              uint3 lid [[thread_position_in_threadgroup]]) {
    const uint hv_idx      = wid.x;
    const uint col         = wid.y * 4u + lid.y;
    const uint lane        = lid.x;
    const uint n_tokens    = p[0];
    const uint qkv_dim     = p[1];
    const uint num_k_heads = p[2];
    const uint num_v_heads = p[3];
    const float q_scale    = as_type<float>(p[4]);
    const uint state_off   = p[5] / 4u;
    const uint q_off       = p[6];
    const uint k_off       = p[7];

    const uint v_off       = 2u * num_k_heads * 128u;
    const uint hk_idx      = hv_idx % num_k_heads;
    const uint state_base  = state_off + hv_idx * (128u * 128u);

    float s_shard[4];
    for (uint r = 0; r < 4u; r++) {
        uint k_row = r * 32u + lane;
        s_shard[r] = state[state_base + k_row * 128u + col];
    }

    const float dt_bias_hv = DO_FUSE_AB ? dt_bias[hv_idx] : 0.0f;
    const float a_log_hv   = DO_FUSE_AB ? a_log[hv_idx]   : 0.0f;

    for (uint t = 0; t < n_tokens; t++) {
        const uint tok_base = t * qkv_dim;
        device const float *q_t = qkv + tok_base + q_off  + hk_idx * 128u;
        device const float *k_t = qkv + tok_base + k_off  + hk_idx * 128u;
        device const float *v_t = qkv + tok_base + v_off  + hv_idx * 128u;
        float decay, b;
        if (DO_FUSE_AB) {
            float raw_a = alpha[t * num_v_heads + hv_idx] + dt_bias_hv;
            float sp    = (raw_a > 20.0f) ? raw_a : log(1.0f + exp(raw_a));
            decay       = exp(sp * a_log_hv);
            float raw_b = beta[t * num_v_heads + hv_idx];
            b           = 1.0f / (1.0f + exp(-raw_b));
        } else {
            decay = alpha[t * num_v_heads + hv_idx];
            b     = beta [t * num_v_heads + hv_idx];
        }

        float k_reg[4];
        float q_reg[4];
        for (uint r = 0; r < 4u; r++) {
            uint k_row = r * 32u + lane;
            s_shard[r] *= decay;
            k_reg[r] = k_t[k_row];
            q_reg[r] = q_t[k_row];
        }

        if (DO_FUSE_L2NORM) {
            float qsq = 0.0f, ksq = 0.0f;
            for (uint r = 0; r < 4u; r++) {
                qsq += q_reg[r] * q_reg[r];
                ksq += k_reg[r] * k_reg[r];
            }
            qsq += simd_shuffle_xor(qsq, 16u);
            qsq += simd_shuffle_xor(qsq, 8u);
            qsq += simd_shuffle_xor(qsq, 4u);
            qsq += simd_shuffle_xor(qsq, 2u);
            qsq += simd_shuffle_xor(qsq, 1u);
            ksq += simd_shuffle_xor(ksq, 16u);
            ksq += simd_shuffle_xor(ksq, 8u);
            ksq += simd_shuffle_xor(ksq, 4u);
            ksq += simd_shuffle_xor(ksq, 2u);
            ksq += simd_shuffle_xor(ksq, 1u);
            float inv_qn = 1.0f / (sqrt(qsq) + 1e-6f);
            float inv_kn = 1.0f / (sqrt(ksq) + 1e-6f);
            for (uint r = 0; r < 4u; r++) {
                q_reg[r] *= inv_qn;
                k_reg[r] *= inv_kn;
            }
        }

        float kv_partial = 0.0f;
        for (uint r = 0; r < 4u; r++)
            kv_partial += s_shard[r] * k_reg[r];
        kv_partial += simd_shuffle_xor(kv_partial, 16u);
        kv_partial += simd_shuffle_xor(kv_partial, 8u);
        kv_partial += simd_shuffle_xor(kv_partial, 4u);
        kv_partial += simd_shuffle_xor(kv_partial, 2u);
        kv_partial += simd_shuffle_xor(kv_partial, 1u);
        float delta = (v_t[col] - kv_partial) * b;

        float attn_partial = 0.0f;
        for (uint r = 0; r < 4u; r++) {
            s_shard[r] += k_reg[r] * delta;
            attn_partial += s_shard[r] * q_reg[r];
        }
        attn_partial += simd_shuffle_xor(attn_partial, 16u);
        attn_partial += simd_shuffle_xor(attn_partial, 8u);
        attn_partial += simd_shuffle_xor(attn_partial, 4u);
        attn_partial += simd_shuffle_xor(attn_partial, 2u);
        attn_partial += simd_shuffle_xor(attn_partial, 1u);
        if (lane == 0) {
            uint out_idx = t * num_v_heads * 128u + hv_idx * 128u + col;
            out[out_idx] = attn_partial * q_scale;
        }
    }

    for (uint r = 0; r < 4u; r++) {
        uint k_row = r * 32u + lane;
        state[state_base + k_row * 128u + col] = s_shard[r];
    }
}
