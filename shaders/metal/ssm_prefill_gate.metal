#include <metal_stdlib>
using namespace metal;

kernel void ssm_prefill_gate(device float       *out    [[buffer(0)]],
                             device const float *z      [[buffer(1)]],
                             device const float *norm_w [[buffer(2)]],
                             constant uint      *p      [[buffer(3)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_sums[8];
    uint hv_idx      = wid.x;
    uint tok         = wid.y;
    uint tid         = lid.x;
    uint head_v_dim  = p[0];
    float eps        = as_type<float>(p[1]);
    uint num_v_heads = p[2];
    uint simd_id     = tid / 32;
    uint simd_lane   = tid % 32;

    if (hv_idx >= num_v_heads) return;

    uint base = (tok * num_v_heads + hv_idx) * head_v_dim;

    float ss = 0.0f;
    float comp = 0.0f;
    for (uint d = tid; d < head_v_dim; d += 256) {
        float val = out[base + d];
        float y = val * val - comp;
        float t = ss + y;
        comp = (t - ss) - y;
        ss = t;
    }

    float partial = simd_sum(ss);
    if (simd_lane == 0) simd_sums[simd_id] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) {
        float v = simd_sums[tid];
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        if (tid == 0) simd_sums[0] = 1.0f / sqrt(v / float(head_v_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = simd_sums[0];

    for (uint d = tid; d < head_v_dim; d += 256) {
        float normed = out[base + d] * inv_rms * norm_w[d];
        float g = z[base + d];
        out[base + d] = normed * (g / (1.0f + exp(-g)));
    }
}
