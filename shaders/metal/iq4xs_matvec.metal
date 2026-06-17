#include <metal_stdlib>
using namespace metal;


constant uint TILE_ROWS = 32;
constant uint THREADS_PER_ROW = 8;
constant uint BLOCK_BYTES = 136;
constant uint QK_K = 256;

constant int kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

kernel void iq4xs_matvec(device const uchar *weights [[buffer(0)]],
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
    uint n_blocks = cols / QK_K;
    uint x_base = token * cols;

    float acc = 0.0f;
    if (global_row < rows) {
        uint row_byte = global_row * n_blocks * BLOCK_BYTES;
        for (uint bi = 0; bi < n_blocks; bi++) {
            device const uchar *block = weights + row_byte + bi * BLOCK_BYTES;
            float d = float(*(device const half *)block);
            uint scales_h = (uint)(*(device const ushort *)(block + 2));
            device const uchar *scales_l = block + 4;
            device const uchar *qs       = block + 8;

            uint j = local_elem;
            int lo = (int)((scales_l[j / 2] >> ((j & 1u) * 4)) & 0xFu);
            int hi = (int)((scales_h >> (j * 2)) & 3u);
            float dl = d * float((lo | (hi << 4)) - 32);

            device const uchar *sub_qs = qs + j * 16;
            uint sub_x_base = x_base + bi * QK_K + j * 32;
            float sub = 0.0f;
            for (uint i = 0; i < 16; i++) {
                uchar q = sub_qs[i];
                int vlo = kvalues_iq4nl[q & 0xFu];
                int vhi = kvalues_iq4nl[q >> 4];
                sub += float(vlo) * x[sub_x_base + i];
                sub += float(vhi) * x[sub_x_base + i + 16];
            }
            acc += dl * sub;
        }
    }

    float val = acc;
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    if (local_elem == 0 && global_row < rows)
        out[out_offset + token * rows + global_row] = val;
}
