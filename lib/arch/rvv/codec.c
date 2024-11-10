#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "../../../include/libbase64.h"
#include "../../tables/tables.h"
#include "../../codecs.h"
#include "config.h"
#include "../../env.h"

#ifdef __riscv
#if HAVE_RVV
#define BASE64_USE_RVV
#endif
#endif

#ifdef BASE64_USE_RVV
#include <riscv_vector.h>
#include "enc_loop_rvv.c"
#include "dec_loop_rvv.c"
#endif // BASE64_USE_RVV

void base64_stream_encode_rvv BASE64_ENC_PARAMS
{
#ifdef BASE64_USE_RVV
#include "../generic/enc_head.c"
	enc_loop_rvv(&s, &slen, &o, &olen);
#include "../generic/enc_tail.c"
#else
	base64_enc_stub(state, src, srclen, out, outlen);
#endif
}

int base64_stream_decode_rvv BASE64_DEC_PARAMS
{
#ifdef BASE64_USE_RVV
#include "../generic/dec_head.c"
	dec_loop_rvv(&s, &slen, &o, &olen);
#include "../generic/dec_tail.c"
#else
	return base64_dec_stub(state, src, srclen, out, outlen);
#endif
}
