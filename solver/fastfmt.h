// fastfmt.h -- exact replacements for snprintf("%g") (precision 6) and "%ld".
// fmt_g: exact for positive finite doubles in [1e-5, 1e5) using 128-bit integer arithmetic
// (round-half-even on the exact binary value, as glibc printf does); everything else falls back to snprintf.
// Digits come from a 1000-entry table of 3-digit groups. Both functions may write up to 8 bytes past the
// returned length: callers reserve room.
#ifndef SWIFTTN93_FASTFMT_H
#define SWIFTTN93_FASTFMT_H

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace fastfmt {

static const uint64_t POW10[20] = {1ULL,
                                   10ULL,
                                   100ULL,
                                   1000ULL,
                                   10000ULL,
                                   100000ULL,
                                   1000000ULL,
                                   10000000ULL,
                                   100000000ULL,
                                   1000000000ULL,
                                   10000000000ULL,
                                   100000000000ULL,
                                   1000000000000ULL,
                                   10000000000000ULL,
                                   100000000000000ULL,
                                   1000000000000000ULL,
                                   10000000000000000ULL,
                                   100000000000000000ULL,
                                   1000000000000000000ULL,
                                   10000000000000000000ULL};

// truncated q = floor(m * 2^e * 10^s) and the remainder test for rounding; s >= 0, e < 0
static inline uint64_t scaled(uint64_t m, int e, int s, int &cmp_half)
{
  unsigned __int128 P = (unsigned __int128)m * POW10[s];
  int sh = -e;
  uint64_t q = (uint64_t)(P >> sh);
  unsigned __int128 rem = P & ((((unsigned __int128)1) << sh) - 1);
  unsigned __int128 half = ((unsigned __int128)1) << (sh - 1);
  cmp_half = rem > half ? 1 : (rem == half ? 0 : -1);
  return q;
}

static inline int fmt_g_slow(char *out, double d) { return snprintf(out, 64, "%g", d); }

// three decimal digits of 0..999 as chars (bytes 0-2) and the number of trailing zero digits (byte 3)
struct Lut3 {
  uint32_t v[1000];
  constexpr Lut3() : v()
  {
    for (int i = 0; i < 1000; i++) {
      int a = i / 100, b = (i / 10) % 10, c = i % 10;
      int tz = i == 0 ? 3 : (c ? 0 : (b ? 1 : 2));
      v[i] = (uint32_t)('0' + a) | (uint32_t)('0' + b) << 8 | (uint32_t)('0' + c) << 16 | (uint32_t)tz << 24;
    }
  }
};
inline constexpr Lut3 LUT3{};

// 10^k as doubles, k = -6..6 (index k + 6); only used to estimate the decade, which is then checked exactly
static const double P10D[13] = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6};

// write %g of d into out (>= 32 bytes available; up to 8 bytes past the returned length may be scribbled);
// kept out of line: inlined into the main loop it cost 10-20% on 2-thread runs of an unrelated set (cause not
// established)
// return number of chars
static inline int fmt_g_emit(char *out, int X, int nd, uint64_t dig);
__attribute__((noinline)) static int fmt_g(char *out, double d)
{
  if (d == 0.0) {
    if (std::signbit(d)) return fmt_g_slow(out, d);
    out[0] = '0';
    return 1;
  }
  if (!(d >= 1e-5 && d < 1e5)) return fmt_g_slow(out, d);
  uint64_t bits;
  memcpy(&bits, &d, 8);
  int be = (int)((bits >> 52) & 0x7ff);
  uint64_t m = (bits & ((1ULL << 52) - 1)) | (1ULL << 52);
  int e = be - 1075;  // d = m * 2^e, e in about [-70, -35]
  // decade: floor((be - 1023) * log10(2)) is X or X - 1; one comparison with 10^(X+1) fixes it (the
  // comparison can be wrong only for the double nearest a power of ten, which the range check below catches)
  int X = ((be - 1023) * 78913) >> 18;  // floor((be - 1023) * log10(2)) up to one unit (arithmetic shift)
  X += d >= P10D[X + 7];
  int cmp = 0;
  uint64_t q = scaled(m, e, 5 - X, cmp);
  if (__builtin_expect(q < 100000ULL || q >= 1000000ULL, 0)) {
    for (int guard = 0; guard < 4; guard++) {
      if (q < 100000ULL)
        X--;
      else if (q >= 1000000ULL)
        X++;
      else
        break;
      int s = 5 - X;
      if (s < 0 || s > 19) return fmt_g_slow(out, d);
      q = scaled(m, e, s, cmp);
    }
    if (q < 100000ULL || q >= 1000000ULL) return fmt_g_slow(out, d);
  }
  q += (cmp > 0 || (cmp == 0 && (q & 1)));  // round half to even on the exact binary value
  if (q == 1000000ULL) {
    q = 100000ULL;
    X++;
  }
  const uint32_t hi = (uint32_t)q / 1000u, lo = (uint32_t)q - hi * 1000u;
  const uint32_t A = LUT3.v[hi], B = LUT3.v[lo];
  const int tz = (B >> 24) == 3 ? 3 + (int)(A >> 24) : (int)(B >> 24);
  const int nd = 6 - tz;  // significant digits printed (>= 1)
  const uint64_t dig = (uint64_t)(A & 0xffffff) | (uint64_t)(B & 0xffffff) << 24;
  return fmt_g_emit(out, X, nd, dig);
}

// the characters of %g from the decimal exponent X (-5..5), the number of significant digits nd and the six
// digit characters (low byte first); same scribbling rules as fmt_g
static inline int fmt_g_emit(char *out, int X, int nd, uint64_t dig)
{
  char dg[8];
  memcpy(dg, &dig, 8);
  if (X < -4) {
    // exponential (only X = -5 reaches here): d.ddddde-05
    char *p = out;
    *p++ = dg[0];
    if (nd > 1) {
      *p++ = '.';
      memcpy(p, dg + 1, 5);
      p += nd - 1;
    }
    int ex = -X;
    *p++ = 'e';
    *p++ = '-';
    *p++ = (char)('0' + ex / 10);
    *p++ = (char)('0' + ex % 10);
    return (int)(p - out);
  }
  if (X < 0) {
    // 0.000ddddd: -X-1 zeros after the point
    memcpy(out, "0.000000", 8);
    char *p = out + 1 - X;
    memcpy(p, dg, 8);
    return (int)(p - out) + nd;
  }
  // X in [0, 5]: X+1 integer digits, then the fraction if any significant digits remain
  const int intd = X + 1;
  memcpy(out, dg, 8);
  if (nd <= intd) return intd;
  out[intd] = '.';
  memcpy(out + intd + 1, dg + intd, 6);
  return nd + 1;
}

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512IFMA__) && defined(__AVX512VL__)
// %g digits for 8 doubles at once (the same exact integer algorithm as fmt_g, in vector form). Returns the
// lanes it handled (positive values in [1e-5, 1e5) whose decade estimate held); for those, X/nd/dig feed
// fmt_g_emit. Other lanes must use fmt_g.
static inline __mmask8 fmt_g8(const double *d8, int64_t Xo[8], int64_t ndo[8], uint64_t digo[8])
{
  const __m512d d = _mm512_loadu_pd(d8);
  __mmask8 ok = _mm512_cmp_pd_mask(d, _mm512_set1_pd(1e-5), _CMP_GE_OQ) & _mm512_cmp_pd_mask(d, _mm512_set1_pd(1e5), _CMP_LT_OQ);
  if (!ok) return 0;
  const __m512i one = _mm512_set1_epi64(1), z = _mm512_setzero_si512();
  const __m512i bits = _mm512_castpd_si512(d);
  const __m512i be = _mm512_and_si512(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(0x7ff));
  const __m512i mm = _mm512_and_si512(bits, _mm512_set1_epi64((1LL << 52) - 1));  // m - 2^52
  __m512i X = _mm512_srai_epi64(_mm512_mullo_epi64(_mm512_sub_epi64(be, _mm512_set1_epi64(1023)), _mm512_set1_epi64(78913)), 18);
  const __m512d p10 = _mm512_mask_i64gather_pd(_mm512_set1_pd(1.0), ok, _mm512_add_epi64(X, _mm512_set1_epi64(7)), P10D, 8);
  X = _mm512_mask_add_epi64(X, _mm512_mask_cmp_pd_mask(ok, d, p10, _CMP_GE_OQ), X, one);
  const __m512i s = _mm512_sub_epi64(_mm512_set1_epi64(5), X);
  ok &= _mm512_cmpge_epi64_mask(s, z) & _mm512_cmple_epi64_mask(s, _mm512_set1_epi64(19));
  const __m512i P = _mm512_mask_i64gather_epi64(z, ok, s, (const long long *)POW10, 8);
  // m * P = (P + hi52(mm * P)) * 2^52 + lo52(mm * P), exact (mm < 2^52, P < 2^35 here)
  const __m512i lo = _mm512_madd52lo_epu64(z, mm, P);
  const __m512i hi = _mm512_madd52hi_epu64(P, mm, P);
  const __m512i sh = _mm512_sub_epi64(_mm512_set1_epi64(1075), be);  // d = m * 2^-sh
  const __mmask8 big = _mm512_cmpge_epi64_mask(sh, _mm512_set1_epi64(52));
  // sh >= 52: t = sh - 52, q = hi >> t, remainder in units of 2^51: R2 = ((hi mod 2^t) << 1) | (lo >> 51)
  // against 2^t, ties broken by the low 51 bits of lo
  const __m512i t = _mm512_sub_epi64(sh, _mm512_set1_epi64(52));
  const __m512i qa = _mm512_srlv_epi64(hi, t);
  const __m512i Rh = _mm512_and_si512(hi, _mm512_sub_epi64(_mm512_sllv_epi64(one, t), one));
  const __m512i R2 = _mm512_or_si512(_mm512_slli_epi64(Rh, 1), _mm512_srli_epi64(lo, 51));
  const __m512i H2 = _mm512_sllv_epi64(one, t);
  const __mmask8 lowz = _mm512_testn_epi64_mask(lo, _mm512_set1_epi64((1LL << 51) - 1));
  const __mmask8 gta = _mm512_cmpgt_epu64_mask(R2, H2) | (_mm512_cmpeq_epi64_mask(R2, H2) & ~lowz);
  const __mmask8 eqa = _mm512_cmpeq_epi64_mask(R2, H2) & lowz;
  // sh < 52: q = (hi << (52 - sh)) | (lo >> sh), remainder lo mod 2^sh against 2^(sh-1)
  const __m512i u = _mm512_sub_epi64(_mm512_set1_epi64(52), sh);
  const __m512i qb = _mm512_or_si512(_mm512_sllv_epi64(hi, u), _mm512_srlv_epi64(lo, sh));
  const __m512i rb = _mm512_and_si512(lo, _mm512_sub_epi64(_mm512_sllv_epi64(one, sh), one));
  const __m512i hb = _mm512_sllv_epi64(one, _mm512_sub_epi64(sh, one));
  const __mmask8 gtb = _mm512_cmpgt_epu64_mask(rb, hb), eqb = _mm512_cmpeq_epi64_mask(rb, hb);
  __m512i q = _mm512_mask_blend_epi64(big, qb, qa);
  const __mmask8 gt = (big & gta) | (~big & gtb), eq = (big & eqa) | (~big & eqb);
  ok &= _mm512_cmpge_epu64_mask(q, _mm512_set1_epi64(100000)) & _mm512_cmplt_epu64_mask(q, _mm512_set1_epi64(1000000));
  if (!ok) return 0;
  // round half to even on the exact value
  const __mmask8 odd = _mm512_test_epi64_mask(q, one);
  q = _mm512_mask_add_epi64(q, gt | (eq & odd), q, one);
  const __mmask8 wrap = _mm512_cmpeq_epi64_mask(q, _mm512_set1_epi64(1000000));
  q = _mm512_mask_mov_epi64(q, wrap, _mm512_set1_epi64(100000));
  X = _mm512_mask_add_epi64(X, wrap, X, one);
  // digits: q = 1000 * h + l
  const __m512i h = _mm512_srli_epi64(_mm512_mul_epu32(q, _mm512_set1_epi64(274877907)), 38);
  const __m512i l = _mm512_sub_epi64(q, _mm512_mullo_epi64(h, _mm512_set1_epi64(1000)));
  const __m512i A = _mm512_cvtepu32_epi64(_mm512_mask_i64gather_epi32(_mm256_setzero_si256(), ok, h, (const int *)LUT3.v, 4));
  const __m512i B = _mm512_cvtepu32_epi64(_mm512_mask_i64gather_epi32(_mm256_setzero_si256(), ok, l, (const int *)LUT3.v, 4));
  const __m512i tzA = _mm512_srli_epi64(A, 24), tzB = _mm512_srli_epi64(B, 24);
  const __mmask8 b3 = _mm512_cmpeq_epi64_mask(tzB, _mm512_set1_epi64(3));
  const __m512i tz = _mm512_mask_add_epi64(tzB, b3, tzA, tzB);  // tzB == 3: 3 + tzA
  const __m512i M24 = _mm512_set1_epi64(0xffffff);
  _mm512_storeu_si512(Xo, X);
  _mm512_storeu_si512(ndo, _mm512_sub_epi64(_mm512_set1_epi64(6), tz));
  _mm512_storeu_si512(digo, _mm512_or_si512(_mm512_and_si512(A, M24), _mm512_slli_epi64(_mm512_and_si512(B, M24), 24)));
  return ok;
}
#endif

static inline int fmt_long(char *out, long v)
{
  if ((unsigned long)v < 1000UL) {
    const uint32_t A = LUT3.v[v];
    const int n = v >= 100 ? 3 : (v >= 10 ? 2 : 1);
    const uint32_t t = A >> (8 * (3 - n));
    memcpy(out, &t, 4);
    return n;
  }
  if ((unsigned long)v < 1000000UL) {
    const uint32_t hi = (uint32_t)v / 1000u, lo = (uint32_t)v - hi * 1000u;
    const uint32_t A = LUT3.v[hi], B = LUT3.v[lo];
    const int n = hi >= 100 ? 3 : (hi >= 10 ? 2 : 1);
    const uint32_t t = A >> (8 * (3 - n));
    memcpy(out, &t, 4);
    memcpy(out + n, &B, 4);
    return n + 3;
  }
  char tmp[24];
  int n = 0;
  unsigned long u;
  bool neg = v < 0;
  u = neg ? (unsigned long)(-(v + 1)) + 1UL : (unsigned long)v;
  do {
    tmp[n++] = (char)('0' + u % 10);
    u /= 10;
  } while (u);
  int k = 0;
  if (neg) out[k++] = '-';
  while (n) out[k++] = tmp[--n];
  return k;
}

}  // namespace fastfmt

#endif
