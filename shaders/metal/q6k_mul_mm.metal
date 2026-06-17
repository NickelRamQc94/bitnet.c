
#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define QK_K          256

typedef struct {
    uchar    ql[QK_K/2];
    uchar    qh[QK_K/4];
    int8_t   scales[QK_K/16];
    half     d;
} block_q6_K;

static_assert(sizeof(block_q6_K) == QK_K/2 + QK_K/4 + QK_K/16 + 2, "wrong q6_K block size");

static inline void dequantize_q6_K(device const block_q6_K * xb,
                                   short il,
                                   thread half4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t   * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = float(d_all) * sc * 32.f;
    const float dl0 = float(d_all) * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uchar shr_h = il>2 ? 2 : 0;
    const uchar shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uchar shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = half(dl0 *  (float)((q       ) & 0xFF) - ml);
        reg[i][1] = half(dl1 *  (float)((q       ) & 0xFF00) - ml);
        reg[i][2] = half(dl2 *  (float)((q       ) & 0xFF0000) - ml);
        reg[i][3] = half(dl3 *  (float)((q       ) & 0xFF000000) - ml);
    }
}

constant int  FC_K_q6       [[function_constant(0)]];
constant bool FC_BC_INP_q6  [[function_constant(1)]];
constant bool FC_BC_OUT_q6  [[function_constant(2)]];
constant bool FC_HAVE_K_q6 = is_function_constant_defined(FC_K_q6);
constant bool FC_HAVE_BC_INP_q6 = is_function_constant_defined(FC_BC_INP_q6);
constant bool FC_HAVE_BC_OUT_q6 = is_function_constant_defined(FC_BC_OUT_q6);

kernel void q6k_mul_mm(
        device const char  * srcA   [[buffer(0)]],
        device const float * srcB   [[buffer(1)]],
        device       float * dst    [[buffer(2)]],
        constant     uint  * p      [[buffer(3)]],
        threadgroup  char  * shmem  [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {

    const int M = (int) p[0];
    const int K = FC_HAVE_K_q6 ? FC_K_q6 : (int) p[1];
    const int N = (int) p[2];

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short nl = QK_K / 16;

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

    const uint nb01 = uint((K / QK_K)) * sizeof(block_q6_K);
    const short offset1 = il0 / nl;

    device const block_q6_K * x = (device const block_q6_K *)(srcA + nb01 * (r0 + lr0)) + offset1;

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
            dequantize_q6_K(x, il, temp_a);

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

        if (FC_HAVE_BC_INP_q6 && !FC_BC_INP_q6) {
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
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
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

    bool full_tile = (FC_HAVE_BC_OUT_q6 && !FC_BC_OUT_q6)
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
