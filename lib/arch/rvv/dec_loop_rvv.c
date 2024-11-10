static const int8_t shift_lut[16] = {
    /* 0 */ 0x00, /* 1 */ 0x00, /* 2 */ 0x3e - 0x2b, /* 3 */ 0x34 - 0x30,
    /* 4 */ 0x00 - 0x41, /* 5 */ 0x0f - 0x50, /* 6 */ 0x1a - 0x61, /* 7 */ 0x29 - 0x70,
    /* 8 */ 0x00, /* 9 */ 0x00, /* a */ 0x00, /* b */ 0x00,
    /* c */ 0x00, /* d */ 0x00, /* e */ 0x00, /* f */ 0x00};

static const uint8_t index_decode[66] = {
    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 18, 17, 16, 22, 21, 20,
    26, 25, 24, 30, 29, 28, 34, 33, 32, 38, 37, 36, 42, 41, 40,
    46, 45, 44, 50, 49, 48, 54, 53, 52, 58, 57, 56, 62, 61, 60,
    66, 65, 64, 70, 69, 68, 74, 73, 72, 78, 77, 76, 82, 81, 80,
    86, 85, 84};

/**
 * creates the indices for the decode gather, if VLEN > 512 bit.
 * Index pattern: 2, 1, 0, 6, 5 ,4, 10, 9, 8, ...
 */
vuint8m1_t createGatherIndexDecode(size_t vl)
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

vuint8m1_t createDecodeIndices(size_t vl)
{
    if (vl <= (512 / 8))
    {
        return __riscv_vle8_v_u8m1(index_decode, vl);
    }
    else
    {
        return createGatherIndexDecode(vl);
    }
}

static BASE64_FORCE_INLINE void dec_loop_rvv(const uint8_t **s, size_t *slen, uint8_t **o, size_t *olen)
{
    size_t vlmax_8 = __riscv_vsetvlmax_e8m4();
    size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    const vuint8m1_t index_vector = createDecodeIndices(vlmax_e8m1);

    vint8m1_t vec_shift_lut = __riscv_vmv_v_x_i8m1(0, vlmax_8);
    vec_shift_lut = __riscv_vle8_v_i8m1(shift_lut, sizeof(shift_lut) / sizeof(shift_lut[0]));

    for (; *slen >= vlmax_8; *slen -= vlmax_8)
    {
        vint8m4_t data_reg = __riscv_vle8_v_i8m4((const signed char *)*s, vlmax_8);

        // data_reg = vector_loochromkup_naive(data_reg, vlmax_8);
        // data_reg = vector_lookup_vrgather(data_reg, vlmax_8);

        size_t vlmax_8 = __riscv_vsetvlmax_e8m4();

        // const vint8m1_t vec_shift_lut = __riscv_vle8_v_i8m2(lookup_vlen8_m2, vlmax_8);

        // extract higher nibble from 8-bit data
        vuint8m4_t higher_nibble = __riscv_vsrl_vx_u8m4(__riscv_vreinterpret_v_i8m4_u8m4(data_reg), 4, vlmax_8);

        // vint8m1_t upper_bound = __riscv_vrgather_vv_i8m1(vec_upper_lut, higher_nibble, vlmax_8);
        // vint8m1_t lower_bound = __riscv_vrgather_vv_i8m1(vec_lower_lut, higher_nibble, vlmax_8);

        // vbool8_t lower = __riscv_vmslt_vv_i8m1_b8(data, lower_bound, vlmax_8);
        // vbool8_t higher = __riscv_vmsgt_vv_i8m1_b8(data, upper_bound, vlmax_8);
        vbool2_t eq = __riscv_vmseq_vx_i8m4_b2(data_reg, 0x2f, vlmax_8);

        // vbool8_t or = __riscv_vmor_mm_b8(lower, higher, vlmax_8);
        // vbool8_t outside = __riscv_vmandn_mm_b8(eq, or, vlmax_8);

        // int error = __riscv_vfirst_m_b8(outside, vlmax_8);

        // if (error != NO_ERROR)
        // {
        //     printf("ERROR!\n");
        // }

        // vint8m1_t shift = __riscv_vrgather_vv_i8m1(vec_shift_lut, higher_nibble, vlmax_8);

        vlmax_8 = __riscv_vsetvlmax_e8m1();
        vint8m4_t shift = __riscv_vcreate_v_i8m1_i8m4(
            __riscv_vrgather_vv_i8m1(vec_shift_lut, __riscv_vget_v_u8m4_u8m1(higher_nibble, 0), vlmax_8),
            __riscv_vrgather_vv_i8m1(vec_shift_lut, __riscv_vget_v_u8m4_u8m1(higher_nibble, 1), vlmax_8),
            __riscv_vrgather_vv_i8m1(vec_shift_lut, __riscv_vget_v_u8m4_u8m1(higher_nibble, 2), vlmax_8),
            __riscv_vrgather_vv_i8m1(vec_shift_lut, __riscv_vget_v_u8m4_u8m1(higher_nibble, 3), vlmax_8));

        vlmax_8 = __riscv_vsetvlmax_e8m4();
        data_reg = __riscv_vadd_vv_i8m4(data_reg, shift, vlmax_8);
        data_reg = __riscv_vadd_vx_i8m4_m(eq, data_reg, -3, vlmax_8);

        // vuint32m1_t packed_data = pack_data(data_reg, vlmax_8);

        size_t vlmax_32 = __riscv_vsetvlmax_e32m4();

        vuint8m4_t convert = __riscv_vreinterpret_v_i8m4_u8m4(data_reg);
        vuint32m4_t data_vector = __riscv_vreinterpret_v_u8m4_u32m4(convert);

        vuint32m4_t ca = __riscv_vand_vx_u32m4(data_vector, 0x003f003f, vlmax_32);
        ca = __riscv_vsll_vx_u32m4(ca, 6, vlmax_32);

        vuint32m4_t db = __riscv_vand_vx_u32m4(data_vector, 0x3f003f00, vlmax_32);
        db = __riscv_vsrl_vx_u32m4(db, 8, vlmax_32);

        vuint32m4_t t0 = __riscv_vor_vv_u32m4(ca, db, vlmax_32);

        vuint32m4_t t1 = __riscv_vsll_vx_u32m4(t0, 12, vlmax_32);
        vuint32m4_t t2 = __riscv_vsrl_vx_u32m4(t0, 16, vlmax_32);

        vuint32m4_t packed_data = __riscv_vor_vv_u32m4(t1, t2, vlmax_32);

        // rearrange elements in vector

        vlmax_8 = __riscv_vsetvlmax_e8m1();

        vuint8m4_t packed_data_e8m4 = __riscv_vreinterpret_v_u32m4_u8m4(packed_data);
        // vuint8m2_t packed_data_e8m2 = convert;
        vuint8m1_t result_0 = __riscv_vrgather_vv_u8m1(__riscv_vget_v_u8m4_u8m1(packed_data_e8m4, 0), index_vector, vlmax_8);
        vuint8m1_t result_1 = __riscv_vrgather_vv_u8m1(__riscv_vget_v_u8m4_u8m1(packed_data_e8m4, 1), index_vector, vlmax_8);
        vuint8m1_t result_2 = __riscv_vrgather_vv_u8m1(__riscv_vget_v_u8m4_u8m1(packed_data_e8m4, 2), index_vector, vlmax_8);
        vuint8m1_t result_3 = __riscv_vrgather_vv_u8m1(__riscv_vget_v_u8m4_u8m1(packed_data_e8m4, 3), index_vector, vlmax_8);

        size_t vl = __riscv_vsetvl_e8m1((vlmax_8 / 4) * 3);

        __riscv_vse8_v_u8m1(*o, result_0, vl);
        *o += (vlmax_8 / 4) * 3;

        __riscv_vse8_v_u8m1(*o, result_1, vl);
        *o += (vlmax_8 / 4) * 3;

        __riscv_vse8_v_u8m1(*o, result_2, vl);
        *o += (vlmax_8 / 4) * 3;

        __riscv_vse8_v_u8m1(*o, result_3, vl);
        *o += (vlmax_8 / 4) * 3;

        *o += vlmax_8 * 4;
        *olen += ((vlmax_8 / 4) * 3) * 4;
    }
}
