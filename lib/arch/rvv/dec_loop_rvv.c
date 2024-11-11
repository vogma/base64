const int8_t shift_lut[16] = {
    /* 0 */ 0x00, /* 1 */ 0x00, /* 2 */ 0x3e - 0x2b, /* 3 */ 0x34 - 0x30,
    /* 4 */ 0x00 - 0x41, /* 5 */ 0x0f - 0x50, /* 6 */ 0x1a - 0x61, /* 7 */ 0x29 - 0x70,
    /* 8 */ 0x00, /* 9 */ 0x00, /* a */ 0x00, /* b */ 0x00,
    /* c */ 0x00, /* d */ 0x00, /* e */ 0x00, /* f */ 0x00};

const uint8_t index_decode[66] = {
    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 18, 17, 16, 22, 21, 20,
    26, 25, 24, 30, 29, 28, 34, 33, 32, 38, 37, 36, 42, 41, 40,
    46, 45, 44, 50, 49, 48, 54, 53, 52, 58, 57, 56, 62, 61, 60,
    66, 65, 64, 70, 69, 68, 74, 73, 72, 78, 77, 76, 82, 81, 80,
    86, 85, 84};

#define NO_ERROR -1

const int8_t LOWER_INVALID_RVV = 1;
const int8_t UPPER_INVALID_RVV = 1;

const int8_t lower_bound_lut_rvv[16] =
    {LOWER_INVALID_RVV, LOWER_INVALID_RVV, 0x2B, 0x30,
     0x41, 0x50, 0x61, 0x70,
     LOWER_INVALID_RVV, LOWER_INVALID_RVV, LOWER_INVALID_RVV, LOWER_INVALID_RVV,
     LOWER_INVALID_RVV, LOWER_INVALID_RVV, LOWER_INVALID_RVV, LOWER_INVALID_RVV};

const int8_t upper_bound_lut_rvv[16] =
    {
        UPPER_INVALID_RVV, UPPER_INVALID_RVV, 0x2b, 0x39,
        0x4f, 0x5a, 0x6f, 0x7a,
        UPPER_INVALID_RVV, UPPER_INVALID_RVV, UPPER_INVALID_RVV, UPPER_INVALID_RVV,
        UPPER_INVALID_RVV, UPPER_INVALID_RVV, UPPER_INVALID_RVV, UPPER_INVALID_RVV};

/**
 * creates the indices for the decode gather, if VLEN > 512 bit.
 * Index pattern: 2, 1, 0, 6, 5 ,4, 10, 9, 8, ...
 */
static BASE64_FORCE_INLINE vuint8m1_t createGatherIndexDecode_rvv(size_t vl)
{
    size_t index_size = vl / 3 + 1;
    uint8_t indices[index_size];

    for (size_t i = 0, j = 0; i < index_size; i++, j += 3)
    {
        indices[j] = 4 * i + 2;
        indices[j + 1] = 4 * i + 1;
        indices[j + 2] = 4 * i;
    }
    return __riscv_vle8_v_u8m1(indices, vl);
}

static BASE64_FORCE_INLINE vuint8m1_t createDecodeIndices_rvv(size_t vl)
{
    if (vl <= (512 / 8))
    {
        return __riscv_vle8_v_u8m1(index_decode, vl);
    }
    else
    {
        return createGatherIndexDecode_rvv(vl);
    }
}

static BASE64_FORCE_INLINE void dec_loop_rvv(const uint8_t **s, size_t *slen, uint8_t **o, size_t *olen)
{
    size_t vlmax_8 = __riscv_vsetvlmax_e8m2();

    if (*slen < vlmax_8)
    {
        return;
    }

    size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    const vuint8m1_t index_vector = createDecodeIndices_rvv(vlmax_e8m1);

    vint8m1_t vec_shift_lut = __riscv_vmv_v_x_i8m1(0, vlmax_8);
    vec_shift_lut = __riscv_vle8_v_i8m1(shift_lut, sizeof(shift_lut) / sizeof(shift_lut[0]));

    const vint8m1_t vec_upper_lut = __riscv_vle8_v_i8m1(upper_bound_lut_rvv, vlmax_e8m1);
    const vint8m1_t vec_lower_lut = __riscv_vle8_v_i8m1(lower_bound_lut_rvv, vlmax_e8m1);

    for (; *slen >= vlmax_8; *slen -= vlmax_8)
    {
        vint8m2_t data_reg = __riscv_vle8_v_i8m2((const signed char *)*s, vlmax_8);

        size_t vlmax_8 = __riscv_vsetvlmax_e8m2();

        // extract higher nibble from 8-bit data
        vuint8m2_t higher_nibble = __riscv_vsrl_vx_u8m2(__riscv_vreinterpret_v_i8m2_u8m2(data_reg), 4, vlmax_8);

        vint8m2_t upper_bound = __riscv_vcreate_v_i8m1_i8m2(
            __riscv_vrgather_vv_i8m1(vec_upper_lut, __riscv_vget_v_u8m2_u8m1(higher_nibble, 0), vlmax_e8m1),
            __riscv_vrgather_vv_i8m1(vec_upper_lut, __riscv_vget_v_u8m2_u8m1(higher_nibble, 1), vlmax_e8m1));

        vint8m2_t lower_bound = __riscv_vcreate_v_i8m1_i8m2(
            __riscv_vrgather_vv_i8m1(vec_lower_lut, __riscv_vget_v_u8m2_u8m1(higher_nibble, 0), vlmax_e8m1),
            __riscv_vrgather_vv_i8m1(vec_lower_lut, __riscv_vget_v_u8m2_u8m1(higher_nibble, 1), vlmax_e8m1));

        vbool4_t lower = __riscv_vmslt_vv_i8m2_b4(data_reg, lower_bound, vlmax_8);
        vbool4_t higher = __riscv_vmsgt_vv_i8m2_b4(data_reg, upper_bound, vlmax_8);
        vbool4_t eq = __riscv_vmseq_vx_i8m2_b4(data_reg, 0x2f, vlmax_8);

        vbool4_t or = __riscv_vmor_mm_b4(lower, higher, vlmax_8);
        vbool4_t outside = __riscv_vmandn_mm_b4(or, eq, vlmax_8);

        int error = __riscv_vfirst_m_b4(outside, vlmax_8);
        if (error != NO_ERROR)
        {
            break;
        }
        vlmax_8 = __riscv_vsetvlmax_e8m1();
        vint8m2_t shift = __riscv_vcreate_v_i8m1_i8m2(__riscv_vrgather_vv_i8m1(vec_shift_lut, __riscv_vget_v_u8m2_u8m1(higher_nibble, 0), vlmax_8), __riscv_vrgather_vv_i8m1(vec_shift_lut, __riscv_vget_v_u8m2_u8m1(higher_nibble, 1), vlmax_8));

        vlmax_8 = __riscv_vsetvlmax_e8m2();
        data_reg = __riscv_vadd_vv_i8m2(data_reg, shift, vlmax_8);
        data_reg = __riscv_vadd_vx_i8m2_m(eq, data_reg, -3, vlmax_8);

        // vuint32m1_t packed_data = pack_data(data_reg, vlmax_8);

        size_t vlmax_32 = __riscv_vsetvlmax_e32m2();

        vuint8m2_t convert = __riscv_vreinterpret_v_i8m2_u8m2(data_reg);
        vuint32m2_t data_vector = __riscv_vreinterpret_v_u8m2_u32m2(convert);

        vuint32m2_t ca = __riscv_vand_vx_u32m2(data_vector, 0x003f003f, vlmax_32);
        ca = __riscv_vsll_vx_u32m2(ca, 6, vlmax_32);

        vuint32m2_t db = __riscv_vand_vx_u32m2(data_vector, 0x3f003f00, vlmax_32);
        db = __riscv_vsrl_vx_u32m2(db, 8, vlmax_32);

        vuint32m2_t t0 = __riscv_vor_vv_u32m2(ca, db, vlmax_32);

        vuint32m2_t t1 = __riscv_vsll_vx_u32m2(t0, 12, vlmax_32);
        vuint32m2_t t2 = __riscv_vsrl_vx_u32m2(t0, 16, vlmax_32);

        vuint32m2_t packed_data = __riscv_vor_vv_u32m2(t1, t2, vlmax_32);

        // rearrange elements in vector

        vlmax_8 = __riscv_vsetvlmax_e8m1();

        vuint8m2_t packed_data_e8m2 = __riscv_vreinterpret_v_u32m2_u8m2(packed_data);
        // vuint8m2_t packed_data_e8m2 = convert;
        vuint8m1_t result_0 = __riscv_vrgather_vv_u8m1(__riscv_vget_v_u8m2_u8m1(packed_data_e8m2, 0), index_vector, vlmax_8);
        vuint8m1_t result_1 = __riscv_vrgather_vv_u8m1(__riscv_vget_v_u8m2_u8m1(packed_data_e8m2, 1), index_vector, vlmax_8);

        size_t vl = __riscv_vsetvl_e8m1((vlmax_8 / 4) * 3);

        __riscv_vse8_v_u8m1(*o, result_0, vl);
        *o += (vlmax_8 / 4) * 3;

        __riscv_vse8_v_u8m1(*o, result_1, vl);
        *o += (vlmax_8 / 4) * 3;

        *s += vlmax_8 * 2;
        *olen += ((vlmax_8 / 4) * 3) * 2;
    }
}