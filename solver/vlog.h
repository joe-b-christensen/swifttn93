// vlog.h -- 8-lane AVX-512 replica of glibc 2.39's x86-64 `log` (__ieee754_log_fma), bit-identical by
// construction: the same algorithm (ARM optimized-routines log, MIT licence), the same table and
// constants (vlog_data.h, dumped from the system libm), and the same sequence of IEEE operations as the
// compiled glibc code (every FMA where glibc's binary has one, separate mul/add elsewhere). IEEE add,
// sub, mul and fma are correctly rounded, so each lane reproduces the scalar result exactly.
// The caller still verifies against the running libm at startup (vlog_selftest) and falls back to scalar
// `log` if anything differs (e.g. a different glibc).
// Lanes must hold positive normal finite x (the caller guarantees x > 0 and checks for subnormal/inf).
#ifndef SWIFTTN93_VLOG_H
#define SWIFTTN93_VLOG_H

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "vlog_data.h"

namespace vlog {

static inline double D(int k)
{
  double d;
  memcpy(&d, &VLOG_DATA[k], 8);
  return d;
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
// x: 8 positive normal doubles
static inline __m512d log8(__m512d x)
{
  const __m512i ix = _mm512_castpd_si512(x);
  // ---- main path
  const __m512i tmp = _mm512_add_epi64(ix, _mm512_set1_epi64((long long)0xc01a000000000000ULL));  // ix - OFF
  const __m512i idx = _mm512_and_si512(_mm512_srli_epi64(tmp, 45), _mm512_set1_epi64(0x7f));
  const __m512i k = _mm512_srai_epi64(tmp, 52);
  const __m512i iz = _mm512_sub_epi64(ix, _mm512_and_si512(tmp, _mm512_set1_epi64((long long)0xfff0000000000000ULL)));
  const __m512i off = _mm512_add_epi64(_mm512_slli_epi64(idx, 1), _mm512_set1_epi64(18));
  const double *base = reinterpret_cast<const double *>(VLOG_DATA);
  const __m512d invc = _mm512_i64gather_pd(off, base, 8);
  const __m512d logc = _mm512_i64gather_pd(_mm512_add_epi64(off, _mm512_set1_epi64(1)), base, 8);
  const __m512d kd = _mm512_cvtepi64_pd(k);
  const __m512d z = _mm512_castsi512_pd(iz);
  const __m512d Ln2hi = _mm512_set1_pd(D(0)), Ln2lo = _mm512_set1_pd(D(1));
  const __m512d A0 = _mm512_set1_pd(D(2)), A1 = _mm512_set1_pd(D(3)), A2 = _mm512_set1_pd(D(4)),
                A3 = _mm512_set1_pd(D(5)), A4 = _mm512_set1_pd(D(6));
  __m512d w = _mm512_fmadd_pd(kd, Ln2hi, logc);                      // kd*Ln2hi + logc
  __m512d r = _mm512_fmadd_pd(z, invc, _mm512_set1_pd(-1.0));        // z*invc - 1
  __m512d p12 = _mm512_fmadd_pd(r, A2, A1);                          // r*A2 + A1
  __m512d hi = _mm512_add_pd(r, w);                                  // r + w
  __m512d r2 = _mm512_mul_pd(r, r);
  __m512d lo = _mm512_add_pd(_mm512_sub_pd(w, hi), r);               // (w - hi) + r
  lo = _mm512_fmadd_pd(kd, Ln2lo, lo);                               // kd*Ln2lo + lo
  __m512d rr2 = _mm512_mul_pd(r, r2);
  __m512d p34 = _mm512_fmadd_pd(r, A4, A3);                          // r*A4 + A3
  __m512d lo2 = _mm512_fmadd_pd(r2, A0, lo);                         // r2*A0 + lo
  __m512d p = _mm512_fmadd_pd(p34, r2, p12);                         // p34*r2 + p12
  __m512d y = _mm512_fmadd_pd(rr2, p, lo2);                          // rr2*p + lo2
  __m512d ymain = _mm512_add_pd(y, hi);
  // ---- close to 1: (ix - LO) < (HI - LO), unsigned
  const __m512i d1 = _mm512_add_epi64(ix, _mm512_set1_epi64((long long)0xc012000000000000ULL));
  const __mmask8 near1 = _mm512_cmple_epu64_mask(d1, _mm512_set1_epi64(0x308ffffffffffLL));
  if (near1) {
    const __m512d B0 = _mm512_set1_pd(D(7)), B1 = _mm512_set1_pd(D(8)), B2 = _mm512_set1_pd(D(9)),
                  B3 = _mm512_set1_pd(D(10)), B4 = _mm512_set1_pd(D(11)), B5 = _mm512_set1_pd(D(12)),
                  B6 = _mm512_set1_pd(D(13)), B7 = _mm512_set1_pd(D(14)), B8 = _mm512_set1_pd(D(15)),
                  B9 = _mm512_set1_pd(D(16)), B10 = _mm512_set1_pd(D(17));
    __m512d rr = _mm512_sub_pd(x, _mm512_set1_pd(1.0));
    __m512d q1 = _mm512_fmadd_pd(rr, B2, B1);
    __m512d q2 = _mm512_fmadd_pd(rr, B5, B4);
    __m512d s2 = _mm512_mul_pd(rr, rr);
    __m512d q3 = _mm512_fmadd_pd(rr, B8, B7);
    q1 = _mm512_fmadd_pd(s2, B3, q1);
    q2 = _mm512_fmadd_pd(s2, B6, q2);
    __m512d s3 = _mm512_mul_pd(rr, s2);
    __m512d t = _mm512_fmadd_pd(s2, B9, q3);
    t = _mm512_fmadd_pd(s3, B10, t);
    t = _mm512_fmadd_pd(t, s3, q2);
    t = _mm512_fmadd_pd(t, s3, q1);
    const __m512d C27 = _mm512_set1_pd(134217728.0);
    __m512d u = _mm512_fmadd_pd(rr, C27, rr);                        // r*2^27 + r
    __m512d rhi = _mm512_fnmadd_pd(C27, rr, u);                      // u - 2^27*r
    __m512d rh2 = _mm512_mul_pd(rhi, rhi);
    __m512d rlo = _mm512_sub_pd(rr, rhi);
    __m512d hi1 = _mm512_fmadd_pd(rh2, B0, rr);                      // rhi^2*B0 + r
    __m512d m = _mm512_sub_pd(rr, hi1);                              // r - hi
    __m512d sum = _mm512_add_pd(rr, rhi);                            // r + rhi
    __m512d lo1 = _mm512_fmadd_pd(rh2, B0, m);                       // rhi^2*B0 + (r - hi)
    __m512d bl = _mm512_mul_pd(B0, rlo);
    lo1 = _mm512_fmadd_pd(bl, sum, lo1);                             // B0*rlo*(r+rhi) + lo
    __m512d y1 = _mm512_fmadd_pd(t, s3, lo1);                        // poly*r3 + lo
    __m512d ynear = _mm512_add_pd(hi1, y1);
    ymain = _mm512_mask_blend_pd(near1, ymain, ynear);
    // exactly 1.0 -> +0.0
    const __mmask8 one = _mm512_cmpeq_epi64_mask(ix, _mm512_set1_epi64(0x3ff0000000000000LL));
    ymain = _mm512_mask_blend_pd(one, ymain, _mm512_setzero_pd());
  }
  return ymain;
}

// lanes whose x is not a positive normal finite number (glibc's special-case branch)
static inline __mmask8 special8(__m512d x)
{
  __m512i top = _mm512_srli_epi64(_mm512_castpd_si512(x), 48);
  __m512i t = _mm512_sub_epi64(top, _mm512_set1_epi64(0x10));
  return _mm512_cmpgt_epu64_mask(t, _mm512_set1_epi64(0x7fdf));
}
#endif

}  // namespace vlog

#endif
