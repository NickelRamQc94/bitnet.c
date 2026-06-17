#include <metal_stdlib>
using namespace metal;


constant uint MAX_HEAD_SIZE = 256;

kernel void prefill_kv_prep(device float       *k        [[buffer(0)]],
                             device float       *v        [[buffer(1)]],
                             device const float *k_bias   [[buffer(2)]],
                             device const float *v_bias   [[buffer(3)]],
                             device const float *k_norm_w [[buffer(4)]],
                             device const float *rope_cos [[buffer(5)]],
                             device const float *rope_sin [[buffer(6)]],
                             constant uint      *p        [[buffer(7)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    uint n_kv_heads      = p[0];
    uint head_size       = p[1];
    uint n_tokens        = p[2];
    uint rope_dims       = p[3];
    uint qk_norm_per_head = p[4];
    uint use_k_bias      = p[5];
    uint use_v_bias      = p[6];
    uint use_k_norm      = p[7];
    uint use_rope        = p[8];
    float eps            = as_type<float>(p[9]);
    uint rope_stride     = p[10];

    uint kv_h = wid.x;
    uint t    = wid.y;
    uint tid  = lid.x;

    if (kv_h >= n_kv_heads || t >= n_tokens) return;
    if (head_size > MAX_HEAD_SIZE) return;

    threadgroup float scratch[256];
    threadgroup float head_scratch[MAX_HEAD_SIZE];

    uint kv_dim = n_kv_heads * head_size;
    uint head_off = t * kv_dim + kv_h * head_size;
    uint bias_off = kv_h * head_size;
    uint norm_off = (qk_norm_per_head != 0u) ? kv_h * head_size : 0u;

    if (use_k_bias != 0u) {
        for (uint i = tid; i < head_size; i += 256u)
            k[head_off + i] = k[head_off + i] + k_bias[bias_off + i];
    }
    if (use_v_bias != 0u) {
        for (uint i = tid; i < head_size; i += 256u)
            v[head_off + i] = v[head_off + i] + v_bias[bias_off + i];
    }

    if (use_k_norm != 0u) {
        threadgroup_barrier(mem_flags::mem_device);

        for (uint i = tid; i < head_size; i += 256u)
            head_scratch[i] = k[head_off + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float ss = 0.0f;
        for (uint d = tid; d < head_size; d += 256u) {
            float v_ = head_scratch[d];
            ss += v_ * v_;
        }
        scratch[tid] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv_rms = 1.0f / sqrt(scratch[0] / float(head_size) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < head_size; d += 256u)
            k[head_off + d] = head_scratch[d] * k_norm_w[norm_off + d] * inv_rms;
    }

    if (use_rope != 0u && rope_dims > 0u) {
        threadgroup_barrier(mem_flags::mem_device);
        uint half_rope = rope_dims / 2u;
        uint rope_row = t * rope_stride;
        for (uint i = tid; i < half_rope; i += 256u) {
            float cos_a = rope_cos[rope_row + i];
            float sin_a = rope_sin[rope_row + i];
            uint i0 = head_off + i;
            uint i1 = i0 + half_rope;
            float v0 = k[i0];
            float v1 = k[i1];
            k[i0] = v0 * cos_a - v1 * sin_a;
            k[i1] = v0 * sin_a + v1 * cos_a;
        }
    }
}
