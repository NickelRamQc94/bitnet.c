#include <metal_stdlib>
using namespace metal;

kernel void ssm_prefill_alpha_beta(device float       *alpha   [[buffer(0)]],
                                   device float       *beta    [[buffer(1)]],
                                   device const float *dt_bias [[buffer(2)]],
                                   device const float *a_log   [[buffer(3)]],
                                   constant uint      *p       [[buffer(4)]],
                                   uint3 wid [[threadgroup_position_in_grid]],
                                   uint3 lid [[thread_position_in_threadgroup]]) {
    uint i           = wid.x * 256 + lid.x;
    uint num_v_heads = p[0];
    uint n_tokens    = p[1];
    uint total       = num_v_heads * n_tokens;
    if (i >= total) return;

    uint hv = i % num_v_heads;
    float dt = alpha[i] + dt_bias[hv];
    float dt_sp = (dt > 20.0f) ? dt : log(1.0f + exp(dt));
    alpha[i] = exp(dt_sp * a_log[hv]);
    beta[i]  = 1.0f / (1.0f + exp(-beta[i]));
}
