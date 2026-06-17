#include <metal_stdlib>
using namespace metal;



template<uint MAX_PREFILL_KV>
kernel void prefill_attn_impl(device const float *q          [[buffer(0)]],
                          device const float *k          [[buffer(1)]],
                          device const float *v          [[buffer(2)]],
                          device       float *out        [[buffer(3)]],
                          device const float *key_cache  [[buffer(4)]],
                          device const float *value_cache[[buffer(5)]],
                          device const float *gate_buf   [[buffer(7)]],
                          constant uint      *p          [[buffer(6)]],
                          uint3 wid [[threadgroup_position_in_grid]],
                          uint3 lid [[thread_position_in_threadgroup]]) {
    uint n_tokens   = p[0];
    uint n_heads    = p[1];
    uint n_kv_heads = p[2];
    uint head_size  = p[3];
    uint kv_mul     = p[4];
    uint pos0       = p[5];
    uint seq_len    = p[6];
    float scale     = as_type<float>(p[7]);
    uint loff_floats = p[8];
    uint kv_dim     = p[9];
    uint use_cache  = p[10];
    uint use_q_gate = p[11];
    (void)n_kv_heads;

    uint h = wid.x;
    uint t = wid.y;
    uint tid     = lid.x;
    uint simd_id = tid >> 5;
    uint lane    = tid & 31;

    uint pos = pos0 + t;
    if (h >= n_heads || t >= n_tokens || (pos + 1u) > MAX_PREFILL_KV) return;

    threadgroup float scores[MAX_PREFILL_KV];
    threadgroup float simd_scratch[8];

    uint kv_h    = h / kv_mul;
    uint q_off   = t * n_heads    * head_size + h    * head_size;
    uint out_off = t * n_heads    * head_size + h    * head_size;
    uint cur_kv_stride = n_kv_heads * head_size;
    uint cur_kv_h_off  = kv_h * head_size;
    uint cache_kv_h_off = kv_h * head_size;

    uint n_kv = pos + 1u;

    for (uint pi = simd_id; pi < n_kv; pi += 8u) {
        uint k_off;
        device const float *k_src;
        if (use_cache != 0u && pi < pos0) {
            uint cyc = pi % seq_len;
            k_off = loff_floats + cyc * kv_dim + cache_kv_h_off;
            k_src = key_cache;
        } else {
            uint t_cur = pi - pos0;
            k_off = t_cur * cur_kv_stride + cur_kv_h_off;
            k_src = k;
        }
        float partial = 0.0f;
        for (uint d = lane; d < head_size; d += 32u)
            partial += q[q_off + d] * k_src[k_off + d];
        float score = simd_sum(partial) * scale;
        if (lane == 0) scores[pi] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -3.402823e+38f;
    for (uint kt = tid; kt < n_kv; kt += 256u)
        local_max = max(local_max, scores[kt]);
    float pmax = simd_max(local_max);
    if (lane == 0) simd_scratch[simd_id] = pmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        float v_ = simd_scratch[tid];
        v_ = max(v_, simd_shuffle_xor(v_, 4));
        v_ = max(v_, simd_shuffle_xor(v_, 2));
        v_ = max(v_, simd_shuffle_xor(v_, 1));
        if (tid == 0) simd_scratch[0] = v_;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_score = simd_scratch[0];

    float local_sum = 0.0f;
    for (uint kt = tid; kt < n_kv; kt += 256u) {
        float e = exp(scores[kt] - max_score);
        scores[kt] = e;
        local_sum += e;
    }
    float psum = simd_sum(local_sum);
    if (lane == 0) simd_scratch[simd_id] = psum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8u) {
        float v_ = simd_scratch[tid];
        v_ += simd_shuffle_xor(v_, 4);
        v_ += simd_shuffle_xor(v_, 2);
        v_ += simd_shuffle_xor(v_, 1);
        if (tid == 0) simd_scratch[0] = v_;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / simd_scratch[0];

    for (uint d = tid; d < head_size; d += 256u) {
        float acc = 0.0f;
        for (uint pi = 0u; pi < n_kv; pi++) {
            uint v_off;
            device const float *v_src;
            if (use_cache != 0u && pi < pos0) {
                uint cyc = pi % seq_len;
                v_off = loff_floats + cyc * kv_dim + cache_kv_h_off;
                v_src = value_cache;
            } else {
                uint t_cur = pi - pos0;
                v_off = t_cur * cur_kv_stride + cur_kv_h_off;
                v_src = v;
            }
            acc += scores[pi] * v_src[v_off + d];
        }
        float result = acc * inv_sum;
        if (use_q_gate != 0u) {
            float g = gate_buf[out_off + d];
            result *= 1.0f / (1.0f + exp(-g));
        }
        out[out_off + d] = result;
    }
}

template [[host_name("prefill_attn")]]
kernel void prefill_attn_impl<4096>(
    device const float *, device const float *, device const float *,
    device float *,       device const float *, device const float *,
    device const float *, constant uint *, uint3, uint3);

template [[host_name("prefill_attn_6144")]]
kernel void prefill_attn_impl<6144>(
    device const float *, device const float *, device const float *,
    device float *,       device const float *, device const float *,
    device const float *, constant uint *, uint3, uint3);
