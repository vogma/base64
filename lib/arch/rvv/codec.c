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
#endif // BASE64_USE_RVV

// Only enable inline assembly on supported compilers.
#if defined(__GNUC__) || defined(__clang__)
#define BASE64_NEON64_USE_ASM
#endif

void base64_stream_encode_rvv BASE64_ENC_PARAMS
{
	base64_enc_stub(state, src, srclen, out, outlen);
}

int base64_stream_decode_rvv BASE64_DEC_PARAMS
{
	return base64_dec_stub(state, src, srclen, out, outlen);
}
