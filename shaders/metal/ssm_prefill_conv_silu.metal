#include <metal_stdlib>
using namespace metal;

kernel void ssm_prefill_conv_silu(device float       *qkv        [[buffer(0)]],
                                  device float       *conv_state [[buffer(1)]],
                                  device const float *conv1d_w   [[buffer(2)]],
                                  constant uint      *p          [[buffer(3)]],
                                  uint3 wid [[threadgroup_position_in_grid]],
                                  uint3 lid [[thread_position_in_threadgroup]]) {
    uint ch       = wid.x * 256 + lid.x;
    uint qkv_dim  = p[0];
    uint kern     = p[1];
    uint cs_off   = p[2];
    uint n_tokens = p[3];

    if (ch >= qkv_dim || kern < 2 || kern > 8) return;

    float hist[7];
    uint hist_n = kern - 1;
    for (uint k = 0; k < hist_n; k++)
        hist[k] = conv_state[cs_off + k * qkv_dim + ch];

    for (uint t = 0; t < n_tokens; t++) {
        float cur = qkv[t * qkv_dim + ch];
        float sum = 0.0f;
        for (uint k = 0; k < hist_n; k++)
            sum += hist[k] * conv1d_w[ch * kern + k];
        sum += cur * conv1d_w[ch * kern + (kern - 1)];
        for (uint k = 0; k + 1 < hist_n; k++)
            hist[k] = hist[k + 1];
        hist[hist_n - 1] = cur;
        qkv[t * qkv_dim + ch] = sum / (1.0f + exp(-sum));
    }

    for (uint k = 0; k < hist_n; k++)
        conv_state[cs_off + k * qkv_dim + ch] = hist[k];
}
