#include <metal_stdlib>
using namespace metal;

kernel void ssm_prefill_delta_precise(
        device float       *state [[buffer(0)]],
        device float       *out   [[buffer(1)]],
        device const float *qkv   [[buffer(2)]],
        device const float *alpha [[buffer(3)]],
        device const float *beta  [[buffer(4)]],
        constant uint      *p     [[buffer(5)]],
        uint3 wid [[threadgroup_position_in_grid]],
        uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float sk[512];
    const uint hv_idx      = wid.x;
    const uint tid         = lid.x;
    const uint n_tokens    = p[0];
    const uint qkv_dim     = p[1];
    const uint num_k_heads = p[2];
    const uint num_v_heads = p[3];
    const float q_scale    = as_type<float>(p[4]);
    const uint state_off   = p[5] / 4u;
    const uint q_off       = p[6];
    const uint k_off       = p[7];

    const uint hk = 128u;
    const uint hv = 128u;
    const uint total = hk * hv;
    const uint v_off = 2u * num_k_heads * hk;
    const uint hk_idx = hv_idx % num_k_heads;
    const uint state_base = state_off + hv_idx * total;

    for (uint t = 0u; t < n_tokens; t++) {
        const uint tok_base = t * qkv_dim;
        device const float *q_t = qkv + tok_base + q_off  + hk_idx * hk;
        device const float *k_t = qkv + tok_base + k_off  + hk_idx * hk;
        device const float *v_t = qkv + tok_base + v_off  + hv_idx * hv;
        const float decay = alpha[t * num_v_heads + hv_idx];
        const float b     = beta [t * num_v_heads + hv_idx];

        for (uint i = tid; i < total; i += 256u)
            state[state_base + i] *= decay;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint vi = tid; vi < hv; vi += 256u) {
            float sum = 0.0f, comp = 0.0f;
            for (uint ki = 0u; ki < hk; ki++) {
                float y = state[state_base + ki * hv + vi] * k_t[ki] - comp;
                float t2 = sum + y;
                comp = (t2 - sum) - y;
                sum = t2;
            }
            sk[vi] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < total; i += 256u) {
            uint ki = i / hv;
            uint vi = i % hv;
            float kk = k_t[ki];
            state[state_base + i] += kk * b * (v_t[vi] - sk[vi]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint vi = tid; vi < hv; vi += 256u) {
            float sum = 0.0f, comp = 0.0f;
            for (uint ki = 0u; ki < hk; ki++) {
                float y = state[state_base + ki * hv + vi] * q_t[ki] - comp;
                float t2 = sum + y;
                comp = (t2 - sum) - y;
                sum = t2;
            }
            uint out_idx = t * num_v_heads * hv + hv_idx * hv + vi;
            out[out_idx] = sum * q_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
