#include <metal_stdlib>
using namespace metal;

kernel void ssm_prefill_l2norm(device float  *qkv [[buffer(0)]],
                               constant uint *p   [[buffer(1)]],
                               uint3 wid [[threadgroup_position_in_grid]],
                               uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float simd_q[4];
    threadgroup float simd_k[4];
    uint head        = wid.x;
    uint tok         = wid.y;
    uint tid         = lid.x;
    uint head_dim    = p[0];
    uint q_off       = p[1];
    uint k_off       = p[2];
    uint num_k_heads = p[3];
    uint qkv_dim     = p[4];
    uint simd_id     = tid / 32;
    uint simd_lane   = tid % 32;

    if (head >= num_k_heads) return;

    uint tok_off = tok * qkv_dim;
    uint qb = tok_off + q_off + head * head_dim;
    uint kb = tok_off + k_off + head * head_dim;

    float qn = 0.0f, qc = 0.0f;
    float kn = 0.0f, kc = 0.0f;
    for (uint d = tid; d < head_dim; d += 128) {
        float qv = qkv[qb + d], kv = qkv[kb + d];
        float qy = qv * qv - qc;
        float qt = qn + qy;
        qc = (qt - qn) - qy;
        qn = qt;
        float ky = kv * kv - kc;
        float kt = kn + ky;
        kc = (kt - kn) - ky;
        kn = kt;
    }

    float partial_q = simd_sum(qn);
    float partial_k = simd_sum(kn);
    if (simd_lane == 0) {
        simd_q[simd_id] = partial_q;
        simd_k[simd_id] = partial_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4) {
        float vq = simd_q[tid];
        float vk = simd_k[tid];
        vq += simd_shuffle_xor(vq, 2);
        vk += simd_shuffle_xor(vk, 2);
        vq += simd_shuffle_xor(vq, 1);
        vk += simd_shuffle_xor(vk, 1);
        if (tid == 0) {
            simd_q[0] = vq;
            simd_k[0] = vk;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_qn = 1.0f / (sqrt(simd_q[0]) + 1e-6f);
    float inv_kn = 1.0f / (sqrt(simd_k[0]) + 1e-6f);

    for (uint d = tid; d < head_dim; d += 128) {
        qkv[qb + d] *= inv_qn;
        qkv[kb + d] *= inv_kn;
    }
}
