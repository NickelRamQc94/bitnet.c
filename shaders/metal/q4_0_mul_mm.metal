
#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define QK_Q4_0 32

static inline void dequantize_q4_0_repacked(device const uchar * nib_block,
                                            float scale,
                                            short il,
                                            thread half4x4 & reg) {
    device const uchar * src = nib_block + (il == 0 ? 0 : 8);
    for (int i = 0; i < 16; ++i) {
        uchar b   = src[i >> 1];
        int   nib = (i & 1) ? int(b >> 4) : int(b & 0xF);
        reg[i/4][i%4] = (half)((float)(nib - 8) * scale);
    }
}

constant int  FC_K_q40       [[function_constant(0)]];
constant bool FC_BC_INP_q40  [[function_constant(1)]];
constant bool FC_BC_OUT_q40  [[function_constant(2)]];
constant bool FC_HAVE_K_q40       = is_function_constant_defined(FC_K_q40);
constant bool FC_HAVE_BC_INP_q40  = is_function_constant_defined(FC_BC_INP_q40);
constant bool FC_HAVE_BC_OUT_q40  = is_function_constant_defined(FC_BC_OUT_q40);

kernel void q4_0_mul_mm(
        device const char  * srcA   [[buffer(0)]],
        device const float * srcB   [[buffer(1)]],
        device       float * dst    [[buffer(2)]],
        constant     uint  * p      [[buffer(3)]],
        threadgroup  char  * shmem  [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {

    const int M = (int) p[0];
    const int K = FC_HAVE_K_q40 ? FC_K_q40 : (int) p[1];
    const int N = (int) p[2];

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = QK_Q4_0 / 16;

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? short(M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? short(N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : short(nr0 - 1);
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : short(nr1 - 1);

    const short il0 = (short)(tiitg % NL0);
    short il = il0;

    const int n_blocks_per_row = K / QK_Q4_0;
    const int n_blocks_total   = M * n_blocks_per_row;
    device const float * scales_all = (device const float *)(srcA);
    device const uchar * nibs_all   = (device const uchar *)(srcA + (size_t)n_blocks_total * sizeof(float));

    const short offset1 = il0 / nl;

    device const float * x_scale = scales_all + (size_t)(r0 + lr0) * n_blocks_per_row + offset1;
    device const uchar * x_nib   = nibs_all   + ((size_t)(r0 + lr0) * n_blocks_per_row + offset1) * 16;

    const short iy = (short)(8 * (tiitg % NL1));
    device const float * y = srcB + (size_t)(r1 + lr1) * K + iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        {
            half4x4 temp_a;
            dequantize_q4_0_repacked(x_nib, *x_scale, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_HAVE_BC_INP_q40 && !FC_BC_INP_q40) {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4*sx + sy;
            using float2x4_t = float2x4;
            using half2x4_t  = half2x4;
            *(threadgroup half2x4_t *)(sb + 64*ib + 8*ly) =
                (half2x4_t)(*((device float2x4_t *) y));
        } else {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg % NL1);
                const short sy = (tiitg / NL1) / 8;
                const short lx = i;
                const short ly = (tiitg / NL1) % 8;
                const short ib = 4*sx + sy;
                *(sb + 64*ib + 8*ly + lx) =
                    (loop_k + iy + i < K) ? (half) y[i] : (half)0;
            }
        }

        il = (il + 2 < nl) ? short(il + 2) : short(il % 2);
        if (il < 2) {
            x_scale += 1;
            x_nib   += 16;
        }
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4*64*(sgitg % 2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg / 2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    bool full_tile = (FC_HAVE_BC_OUT_q40 && !FC_BC_OUT_q40)
                  || (r0 + NR0 <= M && r1 + NR1 <= N);
    if (full_tile) {
        device float * C = dst +
            (r0 + 32*(sgitg & 1)) +
            (r1 + 16*(sgitg >> 1)) * M;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*M*(i/4), M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) +
            32*(sgitg & 1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = dst + r0 + (r1 + j) * M;
                device float4 * D4 = (device float4 *) D;
                threadgroup float  * C  = ((threadgroup float *) shmem) + (j * NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;
                int i = 0;
                for (; i < (nr0/4); i++) D4[i] = C4[i];
                i *= 4;
                for (; i < nr0; i++)     D[i]  = C[i];
            }
        }
    }
}
