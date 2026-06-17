#include <metal_stdlib>
using namespace metal;


constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint ELEMS_PER_THREAD = 256 / THREADS_PER_ROW;
constant uint QK_K = 256;
constant uint BLOCK_BYTES = 176;

static inline uint2 q5k_get_scale_min(uint j, device const uchar *scales) {
    uint sc, m;
    if (j < 4) {
        sc = scales[j] & 63;
        m  = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m  = (scales[j + 4] >> 4)  | ((scales[j] >> 6) << 4);
    }
    return uint2(sc, m);
}

static inline float q5_value(uint qbyte, uint hbyte, uint bit, bool is_high) {
    uint nibble = is_high ? (qbyte >> 4) : (qbyte & 0xF);
    uint hi = (hbyte >> bit) & 1u;
    return float(nibble | (hi << 4));
}

kernel void q5k_matvec_split(device const uchar *weights [[buffer(0)]],
                             device const float *x       [[buffer(1)]],
                             device float       *out0    [[buffer(2)]],
                             device float       *out1    [[buffer(3)]],
                             device float       *out2    [[buffer(4)]],
                             constant uint      *p       [[buffer(5)]],
                             uint3 wid [[threadgroup_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], split1 = p[2], split2 = p[3];
    uint off0 = p[5], off1 = p[6], off2 = p[7];
    uint tile_start = wid.x * TILE_ROWS;
    uint tid = lid.x;

    uint local_row = tid / THREADS_PER_ROW;
    uint local_elem = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;

    uint n_blocks = cols / QK_K;
    uint row_byte = global_row * n_blocks * BLOCK_BYTES;
    uint my_start = local_elem * ELEMS_PER_THREAD;
    uint group = my_start / 64;
    uint is_high_u = (my_start % 64) / 32;
    bool is_high = is_high_u != 0;
    uint bit = group * 2 + is_high_u;
    uint q_off_base = group * 32;

    float acc = 0.0f;
    if (global_row < rows) {
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            float d    = float(*(device const half *)(block));
            float dmin = float(*(device const half *)(block + 2));
            device const uchar *scales = block + 4;
            device const uchar *qh = block + 16;
            device const uchar *qs = block + 48;
            uint elem_base = bi * QK_K;

            uint2 sm = q5k_get_scale_min(bit, scales);
            float ds = d * float(sm.x);
            float dm = dmin * float(sm.y);
            device const uchar *q_off = qs + q_off_base;
            uint xb = elem_base + my_start;

            float sum_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint i = 0; i < 32; i++) {
                float qv = q5_value(q_off[i], qh[i], bit, is_high);
                float xv = x[xb + i];
                sum_qx += qv * xv;
                sum_x += xv;
            }
            acc += ds * sum_qx - dm * sum_x;
        }
    }

    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows) {
        if (split2 > 0u && global_row >= split2) {
            out2[off2 + (global_row - split2)] = val;
        } else if (global_row >= split1) {
            out1[off1 + (global_row - split1)] = val;
        } else {
            out0[off0 + global_row] = val;
        }
    }
}
