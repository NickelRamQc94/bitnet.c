#include <metal_stdlib>
using namespace metal;

static inline float bn_silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void ssm_prefill_silu_gate_stacked(
        device const float *src [[buffer(0)]],
        device float       *dst [[buffer(1)]],
        constant uint      *p   [[buffer(2)]],
        uint3 wid [[threadgroup_position_in_grid]],
        uint3 lid [[thread_position_in_threadgroup]]) {
    uint i        = wid.x * 256u + lid.x;
    uint hidden   = p[0];
    uint n_tokens = p[1];
    if (i >= hidden || wid.y >= n_tokens) return;

    uint tok      = wid.y;
    uint src_base = tok * (2u * hidden);
    float g = src[src_base + i];
    float u = src[src_base + hidden + i];
    dst[tok * hidden + i] = bn_silu(g) * u;
}
