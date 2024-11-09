
static const int8_t offsets[14] = {71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 65};

/**
 * creates gather indices for a given vlen.
 *
 * example vlen=128, bytes shown
 * 0 0 0 0 0 0 0 1 0 0 0 2 0 0 0 3 <- vid 32-bit lanes
 * 0 0 0 0 0 0 1 0 0 0 2 0 0 0 3 0 <- left shift by 8
 * 0 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 <- left shift by 16
 * 0 0 0 0 1 0 0 0 2 0 0 0 3 0 0 0 <- left shift by 24

 * 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 <- or all of the above
 * 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 <- broadcast constant value 3

 * 0 0 0 0 3 3 3 3 6 6 6 6 9 9 9 9 <- multiply both lines above
 * 1 0 2 1 1 0 2 1 1 0 2 1 1 0 2 1 <- broadcast index base value 0x01020001

 * 1 0 2 1 4 3 5 4 7 6 8 7 10 9 11 10 <- add both lines above / finished index values
 */
BASE64_FORCE_INLINE vuint8m1_t createGatherIndexEncode(size_t vl)
{
    vuint32m1_t ids = __riscv_vid_v_u32m1(vl * 4);
    vuint32m1_t ids_shift8 = __riscv_vsll_vx_u32m1(ids, 8, vl);
    vuint32m1_t ids_shift16 = __riscv_vsll_vx_u32m1(ids, 16, vl);
    vuint32m1_t ids_shift24 = __riscv_vsll_vx_u32m1(ids, 24, vl);
    ids = __riscv_vor_vv_u32m1(ids, ids_shift8, vl);
    ids = __riscv_vor_vv_u32m1(ids, ids_shift16, vl);
    ids = __riscv_vor_vv_u32m1(ids, ids_shift24, vl);

    const vuint8m1_t const_vec_3 = __riscv_vmv_v_x_u8m1(3, vl);
    const vuint8m1_t const_index_vec = __riscv_vreinterpret_u8m1(__riscv_vmv_v_x_u32m1(0x01020001, vl));

    vuint8m1_t index_vec = __riscv_vmul_vv_u8m1(const_vec_3, __riscv_vreinterpret_u8m1(ids), vl);

    return __riscv_vadd_vv_u8m1(index_vec, const_index_vec, vl);
}

BASE64_FORCE_INLINE vuint32m4_t lookup_m4(vuint8m4_t data, size_t vl)
{

    const vuint32m4_t const_vec_ac = __riscv_vmv_v_x_u32m4(0x04000040, vl);
    const vuint32m4_t const_vec_bd = __riscv_vmv_v_x_u32m4(0x01000010, vl);

    vuint32m4_t input32 = __riscv_vreinterpret_v_u8m4_u32m4(data);

    // mask out so that only a and c bits remain
    vuint32m4_t index_a_c = __riscv_vand_vx_u32m4(input32, 0x0FC0FC00, vl);

    // mask out so that only a and c bits remain
    vuint32m4_t index_b_d = __riscv_vand_vx_u32m4(input32, 0x003F03F0, vl);

    vl = __riscv_vsetvlmax_e16m4();
    // multiply 16-bit integers and store high 16 bits of 32-bit result
    vuint16m4_t vec_shifted_ac = __riscv_vmulhu_vv_u16m4(__riscv_vreinterpret_v_u32m4_u16m4(index_a_c), __riscv_vreinterpret_v_u32m4_u16m4(const_vec_ac), vl);

    // multiply 16-bit integers and store low 16 bits of 32-bit result
    vuint16m4_t vec_shifted_bd = __riscv_vmul_vv_u16m4(__riscv_vreinterpret_v_u32m4_u16m4(index_b_d), __riscv_vreinterpret_v_u32m4_u16m4(const_vec_bd), vl);

    vl = __riscv_vsetvlmax_e32m4();

    return __riscv_vor_vv_u32m4(__riscv_vreinterpret_v_u16m4_u32m4(vec_shifted_ac), __riscv_vreinterpret_v_u16m4_u32m4(vec_shifted_bd), vl);
}

BASE64_FORCE_INLINE vuint8m4_t table_lookup_m4(vuint8m4_t vec_indices, vint8m1_t offset_vec, size_t vl)
{
    // reduce values 0-64 to 0-13
    vuint8m4_t result = __riscv_vssubu_vx_u8m4(vec_indices, 51, vl);
    vbool2_t vec_lt_26 = __riscv_vmsltu_vx_u8m4_b2(vec_indices, 26, vl);
    const vuint8m4_t vec_lookup = __riscv_vadd_vx_u8m4_mu(vec_lt_26, result, result, 13, vl);

    // shuffle registers one by one
    vint8m1_t offset_vec_0 = __riscv_vrgather_vv_i8m1(offset_vec, __riscv_vget_v_u8m4_u8m1(vec_lookup, 0), vl);
    vint8m1_t offset_vec_1 = __riscv_vrgather_vv_i8m1(offset_vec, __riscv_vget_v_u8m4_u8m1(vec_lookup, 1), vl);
    vint8m1_t offset_vec_2 = __riscv_vrgather_vv_i8m1(offset_vec, __riscv_vget_v_u8m4_u8m1(vec_lookup, 2), vl);
    vint8m1_t offset_vec_3 = __riscv_vrgather_vv_i8m1(offset_vec, __riscv_vget_v_u8m4_u8m1(vec_lookup, 3), vl);

    vint8m4_t offset_vec_bundle = __riscv_vcreate_v_i8m1_i8m4(offset_vec_0, offset_vec_1, offset_vec_2, offset_vec_3);

    vint8m4_t ascii_vec = __riscv_vadd_vv_i8m4(__riscv_vreinterpret_v_u8m4_i8m4(vec_indices), offset_vec_bundle, vl);

    return __riscv_vreinterpret_v_i8m4_u8m4(ascii_vec);
}

static BASE64_FORCE_INLINE void enc_loop_rvv(const uint8_t **s, size_t *slen, uint8_t **o, size_t *olen)
{
    size_t vl;

    size_t vlmax_e8m4 = __riscv_vsetvlmax_e8m4();
    size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    // const vuint8m1_t vec_index_e8m1 = __riscv_vle8_v_u8m1(gather_index_lmul4, vlmax_e8m1);
    const vuint8m1_t vec_index_e8m1 = createGatherIndexEncode(vlmax_e8m1);

    vint8m1_t offset_vec = __riscv_vmv_v_x_i8m1(0, vlmax_e8m1);
    offset_vec = __riscv_vle8_v_i8m1(offsets, (sizeof(offsets) / sizeof(offsets[0])));

    size_t input_slice_e8m4 = (vlmax_e8m4 / 4) * 3;
    size_t input_slice_e8m1 = (vlmax_e8m1 / 4) * 3;

    for (; *slen >= input_slice_e8m4; *slen -= input_slice_e8m4)
    {

        vl = __riscv_vsetvl_e8m1(input_slice_e8m1);

        /**
         * Load (vlmax_e8m1 / 4) * 3 elements into each vector register.
         */
        vuint8m1_t vec_input_0 = __riscv_vle8_v_u8m1(*s, vl);
        *s += (vlmax_e8m1 / 4) * 3;

        vuint8m1_t vec_input_1 = __riscv_vle8_v_u8m1(*s, vl);
        *s += (vlmax_e8m1 / 4) * 3;

        vuint8m1_t vec_input_2 = __riscv_vle8_v_u8m1(*s, vl);
        *s += (vlmax_e8m1 / 4) * 3;

        vuint8m1_t vec_input_3 = __riscv_vle8_v_u8m1(*s, vl);

        vl = __riscv_vsetvl_e8m1(vlmax_e8m1);

        //  the vrgather operation is cheaper at lmul=1 (4*4=16 cycles) than at lmul=4 (64 cycles), therefore each register gets shuffled seperately (https://camel-cdr.github.io/rvv-bench-results/bpi_f3/index.html)
        vuint8m1_t vec_gather_0 = __riscv_vrgather_vv_u8m1(vec_input_0, vec_index_e8m1, vl);
        vuint8m1_t vec_gather_1 = __riscv_vrgather_vv_u8m1(vec_input_1, vec_index_e8m1, vl);
        vuint8m1_t vec_gather_2 = __riscv_vrgather_vv_u8m1(vec_input_2, vec_index_e8m1, vl);
        vuint8m1_t vec_gather_3 = __riscv_vrgather_vv_u8m1(vec_input_3, vec_index_e8m1, vl);

        vuint8m4_t vec_gather = __riscv_vcreate_v_u8m1_u8m4(vec_gather_0, vec_gather_1, vec_gather_2, vec_gather_3);

        vl = __riscv_vsetvlmax_e32m4();

        vuint32m4_t vec_lookup_indices = lookup_m4(vec_gather, vl);

        vl = __riscv_vsetvlmax_e8m4();

        vuint8m4_t base64_chars = table_lookup_m4(__riscv_vreinterpret_v_u32m4_u8m4(vec_lookup_indices), offset_vec, vl);

        __riscv_vse8_v_u8m4((uint8_t *)*o, base64_chars, vl);

        // increase pointer of input data by number of bytes encoded from the input stream
        *s += (vlmax_e8m1 / 4) * 3;

        // increase output pointer by number of bytes produced this round
        *o += vlmax_e8m4;

        // increase total number of bytes produced
        *olen += vlmax_e8m4;
    }
}