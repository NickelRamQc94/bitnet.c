#include <metal_stdlib>
using namespace metal;


constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint BLOCK_BYTES = 110;

static inline int unpack_q3k_scale(uint j, device const uchar *scales) {
    uint lo = (scales[j & 7] >> ((j >> 3) * 4)) & 0xFu;
    uint hi = (scales[8 + (j & 3)] >> ((j >> 2) * 2)) & 0x3u;
    return (int)(lo | (hi << 4)) - 32;
}

kernel void q3k_matvec(device const uchar *weights [[buffer(0)]],
                       device const float *x       [[buffer(1)]],
                       device float       *out     [[buffer(2)]],
                       constant uint      *p       [[buffer(3)]],
                       uint3 wid [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    uint rows = p[0], cols = p[1], extra = p[3];
    uint out_offset = p[5];
    uint tile_start = (extra > 0) ? (wid.x + wid.y * extra) * TILE_ROWS : wid.x * TILE_ROWS;
    uint token = (extra > 0) ? 0 : wid.y;
    uint tid = lid.x;
    uint local_row = tid / THREADS_PER_ROW;
    uint local_elem = tid % THREADS_PER_ROW;
    uint global_row = tile_start + local_row;
    uint n_blocks = cols / 256;
    uint x_base = token * cols;

    float acc = 0.0f;
    if (global_row < rows) {
        uint row_byte = global_row * n_blocks * BLOCK_BYTES;
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            device const uchar *hmask  = block;
            device const uchar *qs     = block + 32;
            device const uchar *scales = block + 96;
            float d = float(*(device const half *)(block + 108));

            uint j_outer = local_elem;
            uint n_sel   = (j_outer >> 2) & 1u;
            uint j       = j_outer & 3u;
            uint shift   = j * 2;
            uint q_off   = n_sel * 32;
            uint hm_bit  = j_outer;
            uint elem_base_in_block = n_sel * 128 + j * 32;

            float dl0 = d * float(unpack_q3k_scale(2 * j_outer + 0, scales));
            float dl1 = d * float(unpack_q3k_scale(2 * j_outer + 1, scales));

            for (uint l = 0; l < 16; l++) {
                int q3 = (int)((qs[q_off + l] >> shift) & 3) -
                         ((hmask[l] & (1u << hm_bit)) ? 0 : 4);
                acc += dl0 * float(q3) *
                       x[x_base + bi * 256 + elem_base_in_block + l];
            }
            for (uint l = 0; l < 16; l++) {
                int q3 = (int)((qs[q_off + 16 + l] >> shift) & 3) -
                         ((hmask[16 + l] & (1u << hm_bit)) ? 0 : 4);
                acc += dl1 * float(q3) *
                       x[x_base + bi * 256 + elem_base_in_block + 16 + l];
            }
        }
    }

    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = val;
}
