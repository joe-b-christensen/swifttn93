// swifttn93 -- exact, fast drop-in replacement for tn93 v1.1.0 (github.com/veg/tn93, commit 076eb0f).
//
// Output (the -o file and the stdout JSON) is byte-identical to `OMP_NUM_THREADS=1 tn93 -q ...` at any
// thread count. Per pair the TN93 distance is a function of the 4x4 table of site-pair counts:
//  * clean/clean sites (A,C,G,T in both) are counted with AVX-512 bit planes, 8 pairs per instruction;
//  * alignment columns where every sequence carries the same clean base are folded into a constant;
//  * ambiguity "events" (a site where either sequence has an IUPAC code) are looked up in tables built by
//    calling tn93's own ambiguity handler (refkernel.cpp). When every contribution a pair receives is a
//    multiple of 1/16 all of tn93's float sums are exact, so the counts can be added in any order;
//  * otherwise (1/3-type contributions, whose rounding depends on site order) the pair goes to tn93's own
//    computeTN93, copied verbatim (refkernel.cpp).
// Reductions that tn93 performs in row order (mean, weights) are done in row order.
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "argparse.hpp"
#include "fastfmt.h"
#include "refkernel.h"
#include "vlog.h"

using namespace std;
using namespace argparse;

#define HISTOGRAM_BINS 200
#define HISTOGRAM_SLICE ((double)HISTOGRAM_BINS)
#define TN93_MAX_DIST 1000.0

#ifdef SWIFT_PROF
#define PROF(x) x
static long prof_slow = 0, prof_fast = 0, prof_fb = 0, prof_skip = 0;
static unsigned long long cyc_k = 0, cyc_s = 0, cyc_t = 0, cyc[8] = {0};
#define TICK(k) do { unsigned long long _t = __rdtsc(); __atomic_add_fetch(&cyc[k], _t - rlast, __ATOMIC_RELAXED); rlast = _t; } while (0)
#include <x86intrin.h>
#else
#define PROF(x)
#endif

//---------------------------------------------------------------- tn93's histogram dump (verbatim)

static void dump_histogram(ostream *outStream, const char *tag, unsigned long *hist)
{
  if (tag) {
    (*outStream) << "\t\"Histogram " << tag << "\" : [";
  } else {
    (*outStream) << "\t\"Histogram\" : [";
  }
  for (unsigned long k = 0; k < HISTOGRAM_BINS; k++) {
    if (k) {
      (*outStream) << ',';
    }
    (*outStream) << '[' << (k + 1.) / HISTOGRAM_BINS << ',' << hist[k] << ']';
  }
  (*outStream) << "]" << endl;
}

//---------------------------------------------------------------- input (exact replica of readFASTA)

// growable byte buffer whose size can be set without zero-filling
struct ByteBuf {
  uint8_t *p = nullptr;
  size_t n = 0, cap = 0;
  ~ByteBuf() { free(p); }
  void reserve(size_t c)
  {
    if (c > cap) {
      // large buffers: 2 MB-aligned with MADV_HUGEPAGE, so first-touch faults are on whole huge pages (in this
      // VM a 4 KB-page fault costs ~0.4 ms per MB written)
      const size_t HP = (size_t)2 << 20;
      uint8_t *q;
      if (c >= 2 * HP) {
        c = (c + HP - 1) / HP * HP;
        q = (uint8_t *)aligned_alloc(HP, c);
        madvise((void *)q, c, MADV_HUGEPAGE);
      } else
        q = (uint8_t *)malloc(c);
      if (n) memcpy(q, p, n);
      free(p);
      p = q;
      cap = c;
    }
  }
  size_t size() const { return n; }
  uint8_t *data() { return p; }
  const uint8_t *data() const { return p; }
  void set_size(size_t k) { n = k; }
};

// fixed-size array that is NOT zero-filled on allocation (every element is written by the parallel builders,
// so pages are first touched by the threads that fill them); 2 MB-aligned with MADV_HUGEPAGE when large
template <class T> struct RVec {
  T *p = nullptr;
  size_t n = 0;
  RVec() = default;
  RVec(const RVec &) = delete;
  RVec &operator=(const RVec &) = delete;
  ~RVec() { free(p); }
  void alloc(size_t k)
  {
    free(p);
    n = k;
    size_t bytes = k * sizeof(T) + 64;
    const size_t HP = (size_t)2 << 20;
    if (bytes >= HP / 2) {  // >= 1 MB: whole 2 MB pages (first-touch faults are expensive in this VM)
      bytes = (bytes + HP - 1) / HP * HP;
      p = (T *)aligned_alloc(HP, bytes);
      madvise((void *)p, bytes, MADV_HUGEPAGE);
    } else
      p = (T *)aligned_alloc(64, (bytes + 63) / 64 * 64);
#ifdef RVEC_POISON  // test builds: garbage instead of zeros, to prove every element read is written first
    memset((void *)p, RVEC_POISON, bytes);
#endif
  }
  T *data() { return p; }
  const T *data() const { return p; }
  size_t size() const { return n; }
  T &operator[](size_t i) { return p[i]; }
  const T &operator[](size_t i) const { return p[i]; }
};

struct SeqSet {
  ByteBuf codes;  // N*L codes 0..17 (ValidChars index)
  vector<char> names;     // concatenated, NUL-terminated
  vector<size_t> name_off;
  vector<long> name_len;
  vector<long> counts;
  long N = 0;
};

static signed char g_tab[256];  // validFlags[toupper(c)]
static unsigned char g_tab128[128];  // same for bytes < 128, invalid = 0xFF

// whole-file read without zero-filling: one read() for regular files, growing buffer otherwise (stdin, pipes)
struct RawBuf {
  unsigned char *p = nullptr;
  size_t n = 0, maplen = 0;
  ~RawBuf()
  {
    if (maplen)
      munmap(p, maplen);
    else
      free(p);
  }
  const unsigned char *data() const { return p; }
  size_t size() const { return n; }
};

static void slurp(FILE *F, RawBuf &B)
{
  int fd = fileno(F);
  struct stat st;
  size_t cap = 1 << 20;
  if (fd >= 0 && fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
    // map the page-cache pages directly (read-only, no copy). The reader never reads past the last byte.
    size_t len = (size_t)st.st_size;
    void *m = mmap(nullptr, len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (m != MAP_FAILED) {
      B.p = (unsigned char *)m;
      B.n = len;
      B.maplen = len;
      return;
    }
    cap = len + 1;
  }
  B.p = (unsigned char *)malloc(cap + 64);
  B.n = 0;
  while (true) {
    if (B.n == cap) {
      cap *= 2;
      B.p = (unsigned char *)realloc(B.p, cap + 64);
    }
    ssize_t r = fd >= 0 ? read(fd, B.p + B.n, cap - B.n) : (ssize_t)fread(B.p + B.n, 1, cap - B.n, F);
    if (r <= 0) break;
    B.n += (size_t)r;
  }
}

static bool vlog_selftest();

// Column statistics of a single input file (constant columns, clean bases seen per column, N count), computed by
// a helper thread on the other core while the reader is still parsing: row i is scanned once the reader has
// accepted it. The helper also runs the vector-log self-test. The reader's code buffer is reserved before
// parsing starts, so row pointers are stable.
struct ColScan {
  std::atomic<long> rows{0};
  std::atomic<bool> done{false};
  const uint8_t *codes = nullptr;
  long L = 0, LW = 0, nN = 0, scanned = 0;
  vector<uint8_t> cst;
  vector<uint64_t> seen;
  bool vlog_ok = false;
  std::thread *th = nullptr;  // never destroyed while running (an exit() during reading just ends the process)
  void scan_row(const uint8_t *r)
  {
    const uint8_t *r0 = codes;
    uint8_t *cs = cst.data();
    uint64_t *sn = seen.data();
    long p = 0, myN = 0;
#if defined(__AVX512BW__)
    for (; p + 64 <= L; p += 64) {
      __m512i v = _mm512_loadu_si512(r + p), f = _mm512_loadu_si512(r0 + p), c = _mm512_loadu_si512(cs + p);
      _mm512_storeu_si512(cs + p, _mm512_maskz_mov_epi8(_mm512_cmpeq_epi8_mask(v, f), c));
      myN += __builtin_popcountll(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(15)));
      uint64_t *sw = sn + (p >> 6);
      sw[0] |= _mm512_cmpeq_epi8_mask(v, _mm512_setzero_si512());
      sw[LW] |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(1));
      sw[2 * LW] |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(2));
      sw[3 * LW] |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(3));
    }
#endif
    for (; p < L; p++) {
      cs[p] &= (uint8_t)(r[p] == r0[p]);
      myN += r[p] == 15;
      if (r[p] < 4) sn[r[p] * LW + (p >> 6)] |= 1ULL << (p & 63);
    }
    nN += myN;
  }
  void run()
  {
    vlog_ok = vlog_selftest();
    long i = 0;
    while (true) {
      const long avail = rows.load(std::memory_order_acquire);
      if (i < avail) {
        for (; i < avail; i++) scan_row(codes + i * L);
        continue;
      }
      if (done.load(std::memory_order_acquire)) {
        if (i >= rows.load(std::memory_order_acquire)) break;
        continue;
      }
      _mm_pause();
    }
    scanned = i;
  }
};
static ColScan &g_cs = *new ColScan;  // never destroyed: an exit() while the helper runs must not free it
static bool g_scan_allowed = false;  // set by main: single input file

static void add_sequence(SeqSet &S, size_t seq_begin, long &firstLen)
{
  long len = (long)(S.codes.size() - seq_begin);
  if (S.N == 0) {
    firstLen = len;
    if (firstLen <= 0) {
      cerr << "First sequence length must be positive." << endl;
      exit(1);
    }
  } else if (len != firstLen) {
    cerr << "All sequences must have the same length (" << firstLen << "), but sequence '"
         << (S.names.data() + S.name_off.back()) << "' had length " << len;
    exit(1);
  }
  S.N++;
  if (g_cs.th)
    g_cs.rows.store(S.N, std::memory_order_release);
#ifndef COLSCAN_MIN_BYTES  // test builds set 0 to exercise the helper on small inputs
#define COLSCAN_MIN_BYTES ((size_t)1 << 20)
#endif
  else if (S.N == 1 && g_scan_allowed && firstLen < 65535 && S.codes.cap >= COLSCAN_MIN_BYTES) {
    g_cs.codes = S.codes.data();
    g_cs.L = firstLen;
    g_cs.LW = (firstLen + 63) / 64;
    g_cs.cst.resize(firstLen);
    for (long p = 0; p < firstLen; p++) g_cs.cst[p] = g_cs.codes[p] < 4;
    g_cs.seen.assign(4 * g_cs.LW, 0);
    g_cs.rows.store(1, std::memory_order_release);
    g_cs.th = new std::thread([] { g_cs.run(); });
  }
}

// returns 0 on success, 1 on a readFASTA error (message printed); exits(1) where tn93 exits.
static int read_fasta(FILE *F, SeqSet &S, long &firstLen, char sep)
{
  RawBuf buf;
  slurp(F, buf);
  const unsigned char *p = buf.data(), *end = p + buf.size();
  int state = 0;
  size_t seq_begin = S.codes.size(), name_begin = S.names.size();
  S.codes.reserve(S.codes.size() + buf.size() + 64);
  while (p < end) {
    if (state == 0) {
      while (p < end && *p != '>' && *p != '#') p++;
      if (p == end) break;
      p++;
      state = 1;
      name_begin = S.names.size();
      continue;
    }
    if (state == 1) {
      const unsigned char *q = p;
      while (q < end && *q != '\n' && *q != '\r') q++;
      S.names.insert(S.names.end(), p, q);
      if (q == end) {
        p = q;
        break;
      }
      p = q + 1;
      long len = (long)(S.names.size() - name_begin);
      S.names.push_back('\0');
      if (len <= 0) {
        cerr << "Sequence names must be non-empty." << endl;
        return 1;
      }
      unsigned long count = 1L;
      if (len >= 3) {
        long sep_loc = 0, ll = (long)S.names.size();
        for (sep_loc = 2; sep_loc < len; sep_loc++) {
          if (S.names[ll - sep_loc - 1] == sep) break;
        }
        if (sep_loc < len) count = atoi(S.names.data() + (ll - sep_loc));
        if (count < 1) count = 1;
      }
      S.counts.push_back((long)count);
      S.name_off.push_back(name_begin);
      S.name_len.push_back(len);
      state = 2;
      seq_begin = S.codes.size();
      continue;
    }
    // state 2: sequence characters (capacity reserved above: at most one code per input byte)
    const unsigned char *q = p;
    {
      size_t old = S.codes.size();
      uint8_t *d0 = S.codes.data() + old, *dst = d0;
#if defined(__AVX512VBMI__) && defined(__AVX512VBMI2__)
      // 64 bytes at a time: map through validFlags[toupper(c)] with a 128-entry byte permute, drop every byte
      // that is not a sequence character (newlines etc.) by compression, stop at '>' or '#'
      {
        const __m512i T0 = _mm512_loadu_si512(g_tab128), T1 = _mm512_loadu_si512(g_tab128 + 64);
        const __m512i gt = _mm512_set1_epi8('>'), hs = _mm512_set1_epi8('#');
        while (q + 64 <= end) {
          __m512i v = _mm512_loadu_si512(q);
          __mmask64 hdr = _mm512_cmpeq_epi8_mask(v, gt) | _mm512_cmpeq_epi8_mask(v, hs);
          __m512i c = _mm512_permutex2var_epi8(T0, v, T1);
          __mmask64 ok = _mm512_cmplt_epu8_mask(c, _mm512_set1_epi8(32)) & _mm512_cmplt_epu8_mask(v, _mm512_set1_epi8((char)128));
          if (hdr) {
            int k = __builtin_ctzll(hdr);
            __mmask64 pre = k ? (~0ULL >> (64 - k)) : 0ULL;
            ok &= pre;
            _mm512_storeu_si512(dst, _mm512_maskz_compress_epi8(ok, c));
            dst += __builtin_popcountll(ok);
            q += k;
            break;
          }
          _mm512_storeu_si512(dst, _mm512_maskz_compress_epi8(ok, c));
          dst += __builtin_popcountll(ok);
          q += 64;
        }
      }
      // remainder (and the bytes before a header found above): skip non-sequence bytes except '>'/'#'
      while (q < end) {
        signed char v = g_tab[*q];
        if (v >= 0) {
          *dst++ = (uint8_t)v;
          q++;
          continue;
        }
        if (*q == '>' || *q == '#') break;
        q++;
      }
#else
      while (q < end) {
        signed char v = g_tab[*q];
        if (v < 0) break;
        *dst++ = (uint8_t)v;
        q++;
      }
#endif
      S.codes.set_size(old + (size_t)(dst - d0));
    }
    p = q;
    if (p == end) break;
    unsigned char c = *p++;
    if (c == '>' || c == '#') {
      add_sequence(S, seq_begin, firstLen);
      state = 1;
      name_begin = S.names.size();
    }
  }
  if (state == 2) {
    add_sequence(S, seq_begin, firstLen);
    return 0;
  }
  cerr << "Unexpected end of file: state " << state << endl;
  return 1;
}

//---------------------------------------------------------------- per-event contribution tables

// One table entry: aggregate contributions of one ambiguity event, scaled by 16 (exact when dyadic).
// lanes: 0 T, 1 D(diag), 2 AG, 3 CT, 4 nfA, 5 nfC, 6 nfG, 7 ambig_count, 8 flags (nondyadic main | avg<<16),
//        9 aT, 10 aD, 11 aAG, 12 aCT, 13 anfA, 14 anfC, 15 anfG   (AVERAGE mode, for the -g fallback)
struct alignas(64) Ent {
  int32_t v[16];
};

static void event_agg(int c1, int c2, int mode, int32_t out[7], int32_t &ambig, bool &dyadic)
{
  double fc[4][4] = {{0.}};
  long ic[4][4] = {{0}};
  unsigned long amb = 0;
  bool skip = false;
  int t1 = c1, t2 = c2;
  if (mode == GAPMM) {
    if (c1 >= GAP && c2 >= GAP)
      skip = true;
    else {
      if (c1 >= GAP) t1 = 15;
      else if (c2 >= GAP) t2 = 15;
    }
  } else if (c1 >= GAP || c2 >= GAP)
    skip = true;
  if (!skip && !(c1 < 4 && c2 < 4)) ref::ambiguity_handler(t1, t2, (char)mode, fc, ic, amb);
  if (!skip && c1 < 4 && c2 < 4) ic[c1][c2]++;
  dyadic = true;
  double cell[4][4];
  for (int a = 0; a < 4; a++)
    for (int b = 0; b < 4; b++) {
      double s = fc[a][b] * 16.0;
      if (s != floor(s)) dyadic = false;
      cell[a][b] = s + 16.0 * (double)ic[a][b];
    }
  for (int k = 0; k < 7; k++) out[k] = 0;
  ambig = (int32_t)amb;
  if (!dyadic) return;
  double T = 0, D = 0, nf[4] = {0, 0, 0, 0};
  for (int a = 0; a < 4; a++)
    for (int b = 0; b < 4; b++) {
      T += cell[a][b];
      nf[a] += cell[a][b];
      nf[b] += cell[a][b];
    }
  for (int a = 0; a < 4; a++) D += cell[a][a];
  out[0] = (int32_t)T;
  out[1] = (int32_t)D;
  out[2] = (int32_t)(cell[0][2] + cell[2][0]);
  out[3] = (int32_t)(cell[1][3] + cell[3][1]);
  out[4] = (int32_t)nf[0];
  out[5] = (int32_t)nf[1];
  out[6] = (int32_t)nf[2];
}

static Ent make_ent(int c1, int c2, int mode)
{
  Ent e;
  memset(&e, 0, sizeof e);
  int32_t m[7], a[7], am, aa;
  bool dm, da;
  event_agg(c1, c2, mode, m, am, dm);
  event_agg(c1, c2, AVERAGE, a, aa, da);
  for (int k = 0; k < 7; k++) e.v[k] = m[k];
  e.v[7] = am;
  e.v[8] = (dm ? 0 : 1) + (da ? 0 : 65536);
  for (int k = 0; k < 7; k++) e.v[9 + k] = a[k];
  return e;
}

//---------------------------------------------------------------- the engine

static inline double tn93_tail(const double nucFreq[4], double fcDiag, double fcAG, double fcCT,
                               unsigned long *histogram, long count1);
static double tn93_tail_ext(const double nucFreq[4], double fcDiag, double fcAG, double fcCT, unsigned long *histogram,
                            long count1)
{
  return tn93_tail(nucFreq, fcDiag, fcAG, fcCT, histogram, count1);
}

static inline bool is_sparse_code(int c, bool useW) { return c >= 4 && c < GAP && !(useW && c == 15); }

struct Engine {
  long N = 0, L = 0, n1 = 0;
  const uint8_t *codes = nullptr;
  int mode = RESOLVE;
  bool fallback_avg = false, fast_ok = true;
  double resolve_fraction = 1.0;
  long min_overlap = 100;
  // compressed (variable) columns
  long Lc = 0, W = 0;
  // bit-plane words (P1/P8) use their own column order: the WV words of truly variable columns first, then
  // the clean-constant columns (at most one clean base in the column, the rest N/gap/ambiguity codes) in
  // words [WV, WP). Two clean bases facing each other there are always equal, so those words need only the
  // clean-overlap and base counts. cc, PNC, BC and the sparse lists keep the original column order.
  long WP = 0, WV = 0;
  long constT = 0, constNf[4] = {0, 0, 0, 0};
  bool useW = false;
  int NP = 4;
  long nblocks = 0;
  RVec<uint64_t> P1, P8;
  RVec<uint8_t> cc;  // N*Lc compressed codes
  vector<uint32_t> sp_off, sp_list;  // packed pos<<5 | code
  vector<Ent> TA, TB;                // [13][18]; TB has a 14th all-zero row for padding
  // block-transposed codes (BC[blk][p][lane]): the row's sparse events against 8 columns in one unrolled pass
  RVec<uint8_t> BC;
  bool avg_force = false, use_bc = false, gapmm = false;
  // words that contain any N (or, under gapmm, gap) bit: per sequence and per 8-column block. The N-plane
  // counts are computed only over the union of the row's and the block's words.
  long WM = 0;
  vector<uint64_t> rowWm, blkWm;
  // clean-constant (K) words: per word, the columns whose clean base has the hi / lo bit; per sequence, how
  // many K columns are non-clean in total and among those masks; per sequence, the K words where it has a
  // non-clean column. For a pair, |both clean| = |K| - n_i - n_j + |nc_i & nc_j| (and likewise per mask),
  // so only the words where the row is non-clean need any work.
  vector<uint64_t> KMhi, KMlo;                // [WP], zero for V words
  long KnT = 0, KnHi = 0, KnLo = 0, KnTT = 0;  // |K|, |K & hi|, |K & lo|, |K & hi & lo|
  RVec<long> kn[4];                           // [nblocks*8] per sequence non-clean counts (T, hi, lo, hi&lo)
  // V words: per sequence clean / hi / lo / hi&lo totals, and the V words where it has a non-clean column. The
  // lane-side counts of a pair (|both clean|, and the lane's hi, lo, hi&lo over the overlap) are these totals
  // minus their intersection with the row's non-clean columns, so kernel8 visits only the row's dirty V words
  // for them.
  RVec<long> vt[4];                           // [nblocks*8]
  vector<uint32_t> vd_off, vd_list;
  vector<uint32_t> kd_off, kd_list;           // per sequence: K words with a non-clean column
  vector<uint64_t> pvalid;                    // [WP] non-padding bits of each plane word
  vector<uint8_t> has3;              // sequence has a B/D/H/V (only used for -a average)
  Ent EW[4], EWW;                    // N vs clean base b, N vs N
  // ordered (site-order replay) path for pairs with non-dyadic contributions
  RVec<uint64_t> PNC;                // [i][w] non-clean non-gap (codes 4..16)
  vector<uint64_t> PNS;              // [i][g] bit w%64 of word g = w/64 set iff PNC[i][w] != 0
  RVec<uint64_t> PNG;                // [i][w] non-gap sites (codes 0..16), original order (nibble replay only)
  bool nib_ok = false;               // no '?' (code 16) in the input and not gapmm: event codes fit 4 bits
  long WS = 0;                       // words per sequence in PNS
  long constCnt[4] = {0, 0, 0, 0};   // constant columns per base
  struct alignas(64) FEnt {
    double fc[16];
    int32_t icd[4];
    int32_t amb;
    int32_t pad[11];
  };
  vector<FEnt> FT[2];                // [main, AVERAGE][c1*18+c2]
  // compact form: every event adds ONE value to a set of cells (tn93's handler adds resolutionsCount, norm or
  // 1/matched to each touched cell), so an entry is (value, 16-bit cell mask, diagonal resolutions, ambig)
  struct alignas(32) CEnt {
    double v;
    uint8_t m0, m1;    // cell mask, low and high byte
    uint16_t amb;
    uint32_t pad;
    uint64_t icd4;     // diagonal resolutions as 4 x 16-bit fields (events per pair < 65535)
    uint64_t pad2;
  };
  vector<CEnt> CT[2];
  vector<CEnt> CT16[2];  // [t][(c1 << 4) | c2], c1, c2 < 16: CT for the nibble-packed replay
  bool compact_ok = true;
  // -a average main path: no table entry has an integer (diagonal) resolution and the ambiguity count is not
  // used (it only feeds the RESOLVE/SUBSET -g test), so the replay loop can skip both sums
  bool lean_avg = false;
  // slow path
  vector<char> mcodes;
  vector<ref::sequence_gap_structure> desc;

  void prepare(const SeqSet &S, long L_, long n1_, int mode_, double rf, long ov)
  {
    N = S.N;
    L = L_;
    n1 = n1_;
    codes = S.codes.data();
    mode = mode_;
    resolve_fraction = rf;
    min_overlap = ov;
    fallback_avg = (mode == RESOLVE || mode == SUBSET) && resolve_fraction < 1.;
    // the x16 aggregates use int32 lanes and 16-bit non-dyadic counters: keep alignments below 65,535 columns
    fast_ok = L < 65535;
    gapmm = (mode == GAPMM);
    if (!fast_ok) {
      // slow-path copies with tn93's own descriptors/jump markers (only -a gapmm uses them)
      mcodes.assign((const char *)codes, (const char *)codes + N * L);
      desc.resize(N);
      bool do_jumps = mode != GAPMM;
      for (long i = 0; i < N; i++) desc[i] = ref::describe_sequence(mcodes.data() + i * L, L, 4UL, do_jumps);
      return;
    }
    // constant columns (every sequence has the same clean base) and N counts, scanned row-wise
    vector<uint8_t> cst;
    long nN = 0;
    const long LW = (L + 63) / 64;
    vector<uint64_t> seen;  // seen[b * LW + p / 64] bit p % 64: some sequence has clean base b at p
    const bool have_scan = g_cs.scanned == N && g_cs.L == L && g_cs.codes == codes;
    if (have_scan) {  // already scanned by the reader's helper thread
      cst.swap(g_cs.cst);
      seen.swap(g_cs.seen);
      nN = g_cs.nN;
    } else {
      cst.resize(L);
      for (long p = 0; p < L; p++) cst[p] = codes[p] < 4;
      seen.assign(4 * LW, 0);
    }
#pragma omp parallel if (!have_scan)
    if (!have_scan) {
      long myN = 0;
      int T = omp_get_num_threads(), t = omp_get_thread_num();
      // 64-column aligned ranges, so each thread owns whole words of `seen`
      long p0 = min(L, LW * t / T * 64), p1 = min(L, LW * (t + 1) / T * 64);
      uint8_t *__restrict cs = cst.data();
      uint64_t *__restrict sn = seen.data();
      const uint8_t *__restrict r0 = codes;
      for (long i = 0; i < N; i++) {
        const uint8_t *__restrict r = codes + i * L;
        long p = p0;
#if defined(__AVX512BW__)
        for (; p + 64 <= p1; p += 64) {
          __m512i v = _mm512_loadu_si512(r + p), f = _mm512_loadu_si512(r0 + p), c = _mm512_loadu_si512(cs + p);
          __mmask64 eq = _mm512_cmpeq_epi8_mask(v, f);
          _mm512_storeu_si512(cs + p, _mm512_maskz_mov_epi8(eq, c));
          myN += __builtin_popcountll(_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(15)));
          uint64_t *sw = sn + (p >> 6);
          sw[0] |= _mm512_cmpeq_epi8_mask(v, _mm512_setzero_si512());
          sw[LW] |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(1));
          sw[2 * LW] |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(2));
          sw[3 * LW] |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(3));
        }
#endif
        for (; p < p1; p++) {
          cs[p] &= (uint8_t)(r[p] == r0[p]);
          myN += r[p] == 15;
          if (r[p] < 4) sn[r[p] * LW + (p >> 6)] |= 1ULL << (p & 63);
        }
      }
#pragma omp atomic
      nN += myN;
    }
    // -a gapmm: a gap facing a non-gap acts as N, so gaps join the N plane (and get their own plane, to
    // exclude gap/gap sites)
    useW = gapmm || nN > 16 * N;
    vector<int> varcol;
    vector<uint64_t> varmask((L + 63) / 64 + 1, 0);  // bit set = variable column
    for (long p = 0; p < L; p++) {
      if (!cst[p]) varmask[p >> 6] |= 1ULL << (p & 63);
      if (cst[p]) {
        uint8_t c0 = codes[p];
        constT++;
        constNf[c0] += 2;
        constCnt[c0]++;
      } else
        varcol.push_back((int)p);
    }
    Lc = (long)varcol.size();
    W = (Lc + 63) / 64;
    // clean-constant variable columns (K) vs the rest (V), as masks over the compressed (cc) words
    vector<uint64_t> Vm(W + 1, 0), Km(W + 1, 0);
    long nV = 0, nK = 0;
    for (long k = 0; k < Lc; k++) {
      long p = varcol[k];
      int nb = 0;
      for (int b = 0; b < 4; b++) nb += (int)(seen[b * LW + (p >> 6)] >> (p & 63) & 1);
      if (nb <= 1) {
        Km[k >> 6] |= 1ULL << (k & 63);
        nK++;
      } else {
        Vm[k >> 6] |= 1ULL << (k & 63);
        nV++;
      }
    }
    WV = (nV + 63) / 64;
    WP = WV + (nK + 63) / 64;
    // hi/lo masks of the K words (plane order: K column number t sits at bit (WV*64 + t))
    KMhi.assign(WP + 1, 0);
    KMlo.assign(WP + 1, 0);
    KnT = nK;
    KnHi = KnLo = KnTT = 0;
    {
      long t = 0;
      for (long k = 0; k < Lc; k++) {
        if (!(Km[k >> 6] >> (k & 63) & 1)) continue;
        long p = varcol[k], q = WV * 64 + t++;
        int base = -1;
        for (int bb = 0; bb < 4; bb++)
          if (seen[bb * LW + (p >> 6)] >> (p & 63) & 1) base = bb;
        if (base >= 0) {
          if (base & 1) KMhi[q >> 6] |= 1ULL << (q & 63), KnHi++;
          if (base & 2) KMlo[q >> 6] |= 1ULL << (q & 63), KnLo++;
          if (base == 3) KnTT++;
        }
      }
    }
    // valid (non-padding) bits of each plane word
    pvalid.assign(WP + 1, 0);
    for (long w = 0; w < WP; w++) {
      long lo = w < WV ? w * 64 : (w - WV) * 64, cnt = w < WV ? nV : nK;
      long nb = max(0L, min(64L, cnt - lo));
      pvalid[w] = nb == 64 ? ~0ULL : ((1ULL << nb) - 1);
    }
    NP = gapmm ? 5 : 4;
    nblocks = (N + 7) / 8;
    long Lpad = W * 64;
    // no zero-fill: the parallel loop below writes every element (P8's lanes past N are cleared here)
    cc.alloc(N * Lc + 64);
    memset(cc.data() + N * Lc, 0, 64);
    P1.alloc(N * WP * NP);
    P8.alloc(nblocks * WP * NP * 8);
    if (N % 8)
      for (long w = 0; w < WP; w++)
        for (int q = 0; q < NP; q++)
          for (long lane = N % 8; lane < 8; lane++) P8[(((nblocks - 1) * WP + w) * NP + q) * 8 + lane] = 0;
    // original-order mask of the sparse sites excluded from sp_list (N, and gaps under gapmm, when useW)
    RVec<uint64_t> XO;
    XO.alloc(N * W);
    sp_off.assign(N + 1, 0);
    PNC.alloc(N * W);
    WS = (W + 63) / 64;
    PNS.assign(N * WS, 0);
    PNG.alloc(N * W);
    std::atomic<bool> has16{false};
    vector<uint32_t> nsp(N, 0);
    const bool allvar = Lc == L;
#pragma omp parallel
    {
      vector<uint8_t> row(Lpad + 64, GAP), prow(WP * 64 + 128, GAP);
#pragma omp for schedule(static)
      for (long i = 0; i < N; i++) {
        const uint8_t *s = codes + i * L;
        uint8_t *d = cc.data() + i * Lc;
        if (allvar)
          memcpy(row.data(), s, L);
        else {
#if defined(__AVX512VBMI2__)
          long k = 0, p = 0;
          for (; p + 64 <= L; p += 64) {
            uint64_t m = varmask[p >> 6];
            _mm512_storeu_si512(row.data() + k, _mm512_maskz_compress_epi8(m, _mm512_loadu_si512(s + p)));
            k += __builtin_popcountll(m);
          }
          for (; p < L; p++)
            if (varmask[p >> 6] >> (p & 63) & 1) row[k++] = s[p];
#else
          for (long k = 0; k < Lc; k++) row[k] = s[varcol[k]];
#endif
        }
        for (long k = Lc; k < Lpad; k++) row[k] = GAP;
        memcpy(d, row.data(), Lc);
        // original order: non-clean sparse sites (PNC) and the N/gap sites excluded from sp_list
        uint32_t ns = 0;
        uint64_t q16 = 0;
        for (long w = 0; w < W; w++) {
          uint64_t wn = 0, nc = 0, gp = 0, ng = 0;
#if defined(__AVX512BW__)
          __m512i v = _mm512_loadu_si512(row.data() + w * 64);
          wn = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(15));
          ng = _mm512_cmplt_epu8_mask(v, _mm512_set1_epi8(GAP));
          nc = _mm512_cmpge_epu8_mask(v, _mm512_set1_epi8(4)) & ng;
          q16 |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(16));
          if (gapmm) gp = _mm512_cmpge_epu8_mask(v, _mm512_set1_epi8(GAP));
#else
          for (int b = 0; b < 64; b++) {
            uint8_t c = row[w * 64 + b];
            uint64_t bit = 1ULL << b;
            if (c == 15) wn |= bit;
            if (c < GAP) ng |= bit;
            if (c == 16) q16 |= bit;
            if (c >= 4 && c < GAP) nc |= bit;
            if (gapmm && c >= GAP) gp |= bit;
          }
#endif
          PNG[i * W + w] = ng;
          if (gapmm) {
            uint64_t valid = (w * 64 + 64 <= Lc) ? ~0ULL : ((1ULL << (Lc - w * 64)) - 1);
            gp &= valid;
            wn |= gp;
            nc |= gp;
          }
          PNC[i * W + w] = nc;
          XO[i * W + w] = useW ? wn : 0;
          if (nc) PNS[i * WS + (w >> 6)] |= 1ULL << (w & 63);
          ns += __builtin_popcountll(useW ? (nc & ~wn) : nc);
        }
        if (q16) has16.store(true, std::memory_order_relaxed);
        nsp[i] = ns;
        // plane order: V columns, then K columns (padding stays GAP)
        {
          uint8_t *pr = prow.data();
#if defined(__AVX512VBMI2__)
          // 64-byte compressed stores padded with GAP; the V pass may spill GAP past WV*64, which the K pass
          // (run second) overwrites; positions no store reaches keep their initial GAP (same for every row)
          const __m512i G = _mm512_set1_epi8(GAP);
          long kv = 0;
          for (long w = 0; w < W; w++) {
            _mm512_storeu_si512(pr + kv, _mm512_mask_compress_epi8(G, Vm[w], _mm512_loadu_si512(row.data() + w * 64)));
            kv += __builtin_popcountll(Vm[w]);
          }
          long kk = WV * 64;
          for (long w = 0; w < W; w++) {
            _mm512_storeu_si512(pr + kk, _mm512_mask_compress_epi8(G, Km[w], _mm512_loadu_si512(row.data() + w * 64)));
            kk += __builtin_popcountll(Km[w]);
          }
#else
          long kv = 0;
          for (long w = 0; w < W; w++)
            for (uint64_t m = Vm[w]; m; m &= m - 1) pr[kv++] = row[w * 64 + __builtin_ctzll(m)];
          for (long k = kv; k < WV * 64; k++) pr[k] = GAP;
          long kk = WV * 64;
          for (long w = 0; w < W; w++)
            for (uint64_t m = Km[w]; m; m &= m - 1) pr[kk++] = row[w * 64 + __builtin_ctzll(m)];
          for (long k = kk; k < WP * 64; k++) pr[k] = GAP;
#endif
        }
        uint64_t *pl = P1.data() + i * WP * NP;
        for (long w = 0; w < WP; w++) {
          uint64_t cm = 0, hi = 0, lo = 0, wn = 0, gp = 0;
#if defined(__AVX512BW__)
          __m512i v = _mm512_loadu_si512(prow.data() + w * 64);
          cm = _mm512_cmplt_epu8_mask(v, _mm512_set1_epi8(4));
          hi = cm & _mm512_test_epi8_mask(v, _mm512_set1_epi8(1));
          lo = cm & _mm512_test_epi8_mask(v, _mm512_set1_epi8(2));
          wn = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(15));
          if (gapmm) gp = _mm512_cmpge_epu8_mask(v, _mm512_set1_epi8(GAP));
#else
          for (int b = 0; b < 64; b++) {
            uint8_t c = prow[w * 64 + b];
            uint64_t bit = 1ULL << b;
            if (c < 4) {
              cm |= bit;
              if (c & 1) hi |= bit;
              if (c & 2) lo |= bit;
            }
            if (c == 15) wn |= bit;
            if (gapmm && c >= GAP) gp |= bit;
          }
#endif
          if (gapmm) {
            gp &= pvalid[w];
            wn |= gp;
            pl[w * NP + 4] = gp;
          }
          pl[w * NP + 0] = hi;
          pl[w * NP + 1] = lo;
          pl[w * NP + 2] = cm;
          pl[w * NP + 3] = useW ? wn : 0;
        }
        long b = i >> 3, lane = i & 7;
        for (long w = 0; w < WP; w++)
          for (int q = 0; q < NP; q++) P8[((b * WP + w) * NP + q) * 8 + lane] = pl[w * NP + q];
      }
    }
    // per-sequence K-word statistics
    for (int q = 0; q < 4; q++) {
      kn[q].alloc(nblocks * 8);
      for (long i = N; i < nblocks * 8; i++) kn[q][i] = 0;
    }
    kd_off.assign(N + 1, 0);
    {
      vector<uint32_t> cntd(N, 0);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < N; i++) {
        long c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        uint32_t nd = 0;
        for (long w = WV; w < WP; w++) {
          uint64_t nc = pvalid[w] & ~P1[(i * WP + w) * NP + 2];
          if (!nc) continue;
          nd++;
          c0 += __builtin_popcountll(nc);
          c1 += __builtin_popcountll(nc & KMhi[w]);
          c2 += __builtin_popcountll(nc & KMlo[w]);
          c3 += __builtin_popcountll(nc & KMhi[w] & KMlo[w]);
        }
        kn[0][i] = c0;
        kn[1][i] = c1;
        kn[2][i] = c2;
        kn[3][i] = c3;
        cntd[i] = nd;
      }
      for (long i = 0; i < N; i++) kd_off[i + 1] = kd_off[i] + cntd[i];
      kd_list.assign(kd_off[N] + 1, 0);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < N; i++) {
        uint32_t o = kd_off[i];
        for (long w = WV; w < WP; w++)
          if (pvalid[w] & ~P1[(i * WP + w) * NP + 2]) kd_list[o++] = (uint32_t)w;
      }
    }
    // per-sequence V-word totals and dirty V words
    for (int q = 0; q < 4; q++) {
      vt[q].alloc(nblocks * 8);
      for (long i = N; i < nblocks * 8; i++) vt[q][i] = 0;
    }
    vd_off.assign(N + 1, 0);
    {
      vector<uint32_t> cntd(N, 0);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < N; i++) {
        long c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        uint32_t nd = 0;
        const uint64_t *pl = P1.data() + i * WP * NP;
        for (long w = 0; w < WV; w++) {
          const uint64_t hi = pl[w * NP], lo = pl[w * NP + 1], cm = pl[w * NP + 2];
          c0 += __builtin_popcountll(cm);
          c1 += __builtin_popcountll(hi);
          c2 += __builtin_popcountll(lo);
          c3 += __builtin_popcountll(hi & lo);
          nd += (pvalid[w] & ~cm) != 0;
        }
        vt[0][i] = c0;
        vt[1][i] = c1;
        vt[2][i] = c2;
        vt[3][i] = c3;
        cntd[i] = nd;
      }
      for (long i = 0; i < N; i++) vd_off[i + 1] = vd_off[i] + cntd[i];
      vd_list.assign(vd_off[N] + 1, 0);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < N; i++) {
        uint32_t o = vd_off[i];
        for (long w = 0; w < WV; w++)
          if (pvalid[w] & ~P1[(i * WP + w) * NP + 2]) vd_list[o++] = (uint32_t)w;
      }
    }
    WM = (WP + 63) / 64;
    rowWm.assign(N * WM + 1, 0);
    blkWm.assign(nblocks * WM + 1, 0);
    if (useW)
      for (long i = 0; i < N; i++)
        for (long w = 0; w < WP; w++)
          if (P1[(i * WP + w) * NP + 3]) {
            rowWm[i * WM + (w >> 6)] |= 1ULL << (w & 63);
            blkWm[(i >> 3) * WM + (w >> 6)] |= 1ULL << (w & 63);
          }
    for (long i = 0; i < N; i++) sp_off[i + 1] = sp_off[i] + nsp[i];
    nib_ok = !gapmm && !has16.load();
    sp_list.assign(sp_off[N], 0);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
      uint32_t o = sp_off[i];
      const uint8_t *d = cc.data() + i * Lc;
      for (long w = 0; w < W; w++) {
        uint64_t m = PNC[i * W + w] & ~XO[i * W + w];
        while (m) {
          long k = w * 64 + __builtin_ctzll(m);
          m &= m - 1;
          sp_list[o++] = ((uint32_t)k << 5) | d[k];
        }
      }
    }
    avg_force = (mode == AVERAGE || mode == GAPMM);  // both average every ambiguity
    has3.assign(N, 0);
    for (long i = 0; i < N; i++)
      for (uint32_t q = sp_off[i]; q < sp_off[i + 1]; q++) {
        uint32_t c = sp_list[q] & 31;
        if (c >= 11 && c <= 14) has3[i] = 1;
      }
    TA.resize(13 * 18);
    TB.assign(14 * 18, Ent());
    memset(TB.data(), 0, TB.size() * sizeof(Ent));
    // block-transposed codes, only worth building when sequences carry sparse codes (>= 1 per sequence)
    use_bc = sp_off[N] >= (uint32_t)N;
    // position-major: BC[p * Npad + j]; a row's sparse sites sweep their position rows sequentially as the
    // column blocks advance, which the hardware prefetcher follows
    const long Npad = nblocks * 8;
    if (use_bc) BC.alloc((size_t)Npad * Lc);
#pragma omp parallel for schedule(static) if (use_bc)
    for (long k0 = 0; k0 < (use_bc ? Lc : 0); k0 += 64) {
      for (long i = 0; i < N; i++) {
        const uint8_t *d = cc.data() + i * Lc;
        for (long k = k0; k < min(Lc, k0 + 64); k++) BC[(size_t)k * Npad + i] = d[k];
      }
      for (long k = k0; k < min(Lc, k0 + 64); k++)
        for (long i = N; i < Npad; i++) BC[(size_t)k * Npad + i] = GAP;
    }
    for (int c = 4; c <= 16; c++)
      for (int o = 0; o < 18; o++) {
        TA[(c - 4) * 18 + o] = make_ent(c, o, mode);  // s1 sparse c, s2 other o
        if (is_sparse_code(o, useW))
          memset(&TB[(c - 4) * 18 + o], 0, sizeof(Ent));
        else
          TB[(c - 4) * 18 + o] = make_ent(o, c, mode);  // s2 sparse c, s1 other o
      }
    for (int b = 0; b < 4; b++) EW[b] = make_ent(15, b, mode);
    EWW = make_ent(15, 15, mode);
    for (int t = 0; t < 2; t++) {
      int m = t == 0 ? mode : AVERAGE;
      FT[t].assign(18 * 18, FEnt());
      for (int c1 = 0; c1 < 18; c1++)
        for (int c2 = 0; c2 < 18; c2++) {
          FEnt &f = FT[t][c1 * 18 + c2];
          memset(&f, 0, sizeof f);
          if (c1 < 4 && c2 < 4) continue;
          bool skip = false;
          int t1 = c1, t2 = c2;
          if (m == GAPMM) {
            if (c1 >= GAP && c2 >= GAP)
              skip = true;
            else if (c1 >= GAP)
              t1 = 15;
            else if (c2 >= GAP)
              t2 = 15;
          } else if (c1 >= GAP || c2 >= GAP)
            skip = true;
          if (skip) continue;
          double fc[4][4] = {{0.}};
          long ic[4][4] = {{0}};
          unsigned long amb = 0;
          ref::ambiguity_handler(t1, t2, (char)m, fc, ic, amb);
          for (int q = 0; q < 16; q++) f.fc[q] = fc[q >> 2][q & 3];
          for (int q = 0; q < 4; q++) f.icd[q] = (int32_t)ic[q][q];
          f.amb = (int32_t)amb;
        }
      CT[t].assign(18 * 18, CEnt());
      for (int k = 0; k < 18 * 18; k++) {
        const FEnt &f = FT[t][k];
        CEnt &e = CT[t][k];
        memset(&e, 0, sizeof e);
        uint16_t mask = 0;
        for (int q = 0; q < 16; q++)
          if (f.fc[q] != 0.) {
            if (mask && f.fc[q] != e.v) compact_ok = false;
            e.v = f.fc[q];
            mask |= (uint16_t)(1u << q);
          }
        e.m0 = (uint8_t)(mask & 0xff);
        e.m1 = (uint8_t)(mask >> 8);
        e.icd4 = 0;
        for (int q = 0; q < 4; q++) {
          if (f.icd[q] < 0 || f.icd[q] > 1) compact_ok = false;
          e.icd4 |= (uint64_t)f.icd[q] << (16 * q);
        }
        if (f.amb < 0 || f.amb > 1) compact_ok = false;
        e.amb = (uint16_t)f.amb;
      }
      if (t == 1) {
        lean_avg = mode == AVERAGE;
        for (int k = 0; k < 18 * 18; k++)
          if (CT[0][k].icd4) lean_avg = false;
      }
      CT16[t].assign(256, CEnt());
      for (int c1 = 0; c1 < 16; c1++)
        for (int c2 = 0; c2 < 16; c2++) CT16[t][(c1 << 4) | c2] = CT[t][c1 * 18 + c2];
    }
  }

  // Site-order replay of every ambiguity event, then tn93's own final merge and formula: exact for any
  // contribution values. cells = clean/clean counts over variable columns (x*4+y, x from seq i).
  // Site-order replay of every ambiguity event of pair (i,j) into fc (16 doubles) and ic (16 longs, clean/clean +
  // resolutions). Gap partners have all-zero table entries, so the walk needs no gap mask.
  inline void replay(long i, long j, const long *cells, int t, double *fc, long *ic, unsigned long &ambig_count) const
  {
#if defined(__AVX512F__) && defined(__AVX512VBMI2__)
    if (compact_ok) {
      if (nib_ok) {
        if (t == 0 && lean_avg)
          replay_nib<false>(i, j, cells, t, fc, ic, ambig_count);
        else
          replay_nib<true>(i, j, cells, t, fc, ic, ambig_count);
      }
      else
        replay_compact(i, j, cells, t, fc, ic, ambig_count);
      return;
    }
#endif
    for (int q = 0; q < 16; q++) ic[q] = cells[q];
    for (int x = 0; x < 4; x++) ic[x * 5] += constCnt[x];
    ambig_count = 0;
    const uint64_t *nc1 = PNC.data() + i * W, *nc2 = PNC.data() + j * W;
    const uint8_t *s1 = cc.data() + i * Lc, *s2 = cc.data() + j * Lc;
    const FEnt *F = FT[t].data();
#if defined(__AVX512F__)
    __m512d f0 = _mm512_setzero_pd(), f1 = _mm512_setzero_pd();
#else
    for (int q = 0; q < 16; q++) fc[q] = 0.;
#endif
    long i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    unsigned long am = 0;
    for (long ww = 0; ww < W; ww++) {
      uint64_t E = nc1[ww] | nc2[ww];
      while (E) {
        long p = ww * 64 + __builtin_ctzll(E);
        E &= E - 1;
        const FEnt &f = F[s1[p] * 18 + s2[p]];
#if defined(__AVX512F__)
        f0 = _mm512_add_pd(f0, _mm512_loadu_pd(f.fc));
        f1 = _mm512_add_pd(f1, _mm512_loadu_pd(f.fc + 8));
#else
        for (int q = 0; q < 16; q++) fc[q] += f.fc[q];
#endif
        i0 += f.icd[0];
        i1 += f.icd[1];
        i2 += f.icd[2];
        i3 += f.icd[3];
        am += f.amb;
      }
    }
#if defined(__AVX512F__)
    _mm512_storeu_pd(fc, f0);
    _mm512_storeu_pd(fc + 8, f1);
#endif
    ic[0] += i0;
    ic[5] += i1;
    ic[10] += i2;
    ic[15] += i3;
    ambig_count = am;
  }

#if defined(__AVX512F__) && defined(__AVX512VBMI2__)
  // branch-free event extraction (VBMI2 byte compress of both sequences' codes under the event mask), then one
  // masked add of the event's value per event, in site order
  inline void replay_compact(long i, long j, const long *cells, int t, double *fc, long *ic,
                             unsigned long &ambig_count) const
  {
    for (int q = 0; q < 16; q++) ic[q] = cells[q];
    for (int x = 0; x < 4; x++) ic[x * 5] += constCnt[x];
    const uint64_t *nc1 = PNC.data() + i * W, *nc2 = PNC.data() + j * W;
    const uint8_t *s1 = cc.data() + i * Lc, *s2 = cc.data() + j * Lc;
    // per-thread scratch (alloca cost a stack-probe loop per call); W <= 1024 on this path (L < 65535)
    static thread_local uint8_t tb1[1024 * 64 + 128], tb2[1024 * 64 + 128];
    uint8_t *b1 = tb1, *b2 = tb2;
    long n = 0;
    // only the words where either sequence has an event (PNS summary bits)
    const uint64_t *ns1 = PNS.data() + i * WS, *ns2 = PNS.data() + j * WS;
    for (long g = 0; g < WS; g++)
      for (uint64_t S = ns1[g] | ns2[g]; S; S &= S - 1) {
        const long ww = g * 64 + __builtin_ctzll(S);
        uint64_t E = nc1[ww] | nc2[ww];
        __m512i v1 = _mm512_loadu_si512(s1 + ww * 64), v2 = _mm512_loadu_si512(s2 + ww * 64);
        _mm512_storeu_si512(b1 + n, _mm512_maskz_compress_epi8(E, v1));
        _mm512_storeu_si512(b2 + n, _mm512_maskz_compress_epi8(E, v2));
        n += __builtin_popcountll(E);
      }
    const CEnt *C = CT[t].data();
    __m512d f0 = _mm512_setzero_pd(), f1 = _mm512_setzero_pd();
    uint64_t ip = 0;
    unsigned long am = 0;
    for (long k = 0; k < n; k++) {
      const CEnt &e = C[b1[k] * 18 + b2[k]];
      __m512d v = _mm512_set1_pd(e.v);
      f0 = _mm512_mask_add_pd(f0, (__mmask8)e.m0, f0, v);
      f1 = _mm512_mask_add_pd(f1, (__mmask8)e.m1, f1, v);
      ip += e.icd4;
      am += e.amb;
    }
    _mm512_storeu_pd(fc, f0);
    _mm512_storeu_pd(fc + 8, f1);
    ic[0] += (long)(ip & 0xffff);
    ic[5] += (long)((ip >> 16) & 0xffff);
    ic[10] += (long)((ip >> 32) & 0xffff);
    ic[15] += (long)(ip >> 48);
    ambig_count = am;
  }

  // Same walk with both codes of an event packed in one byte (c1 << 4 | c2): one byte compress per word instead of
  // two. Valid when no code is 16 ('?') and gaps contribute nothing (not gapmm): events are then the non-clean sites
  // of either sequence where neither is a gap (a gap partner's table entry is all zero, so dropping it is exact).
  template <bool INT>
  inline void replay_nib(long i, long j, const long *cells, int t, double *fc, long *ic,
                         unsigned long &ambig_count) const
  {
    for (int q = 0; q < 16; q++) ic[q] = cells[q];
    for (int x = 0; x < 4; x++) ic[x * 5] += constCnt[x];
    const uint64_t *nc1 = PNC.data() + i * W, *nc2 = PNC.data() + j * W;
    const uint64_t *ng1 = PNG.data() + i * W, *ng2 = PNG.data() + j * W;
    const uint8_t *s1 = cc.data() + i * Lc, *s2 = cc.data() + j * Lc;
    static thread_local uint8_t tb[1024 * 64 + 128];
    uint8_t *bb = tb;
    long n = 0;
    const __m512i F0 = _mm512_set1_epi8((char)0xF0);
    const uint64_t *ns1 = PNS.data() + i * WS, *ns2 = PNS.data() + j * WS;
    for (long g = 0; g < WS; g++)
      for (uint64_t S = ns1[g] | ns2[g]; S; S &= S - 1) {
        const long ww = g * 64 + __builtin_ctzll(S);
        const uint64_t E = (nc1[ww] | nc2[ww]) & ng1[ww] & ng2[ww];
        __m512i v1 = _mm512_loadu_si512(s1 + ww * 64), v2 = _mm512_loadu_si512(s2 + ww * 64);
        // (v1 << 4) & 0xF0 | v2 & 0x0F per byte (16-bit shift; the bits it carries across bytes are masked off)
        __m512i pk = _mm512_ternarylogic_epi32(_mm512_slli_epi16(v1, 4), F0, v2, 0xE2);
        _mm512_storeu_si512(bb + n, _mm512_maskz_compress_epi8(E, pk));
        n += __builtin_popcountll(E);
      }
    const CEnt *C = CT16[t].data();
    __m512d f0 = _mm512_setzero_pd(), f1 = _mm512_setzero_pd();
    uint64_t ip = 0;
    unsigned long am = 0;
    for (long k = 0; k < n; k++) {
      const CEnt &e = C[bb[k]];
      __m512d v = _mm512_set1_pd(e.v);
      // masks straight from memory (kmovb k, m8: a load, not a port-5 GPR-to-mask move)
      __mmask8 k0, k1;
      asm("kmovb %1, %0" : "=k"(k0) : "m"(e.m0));
      asm("kmovb %1, %0" : "=k"(k1) : "m"(e.m1));
      f0 = _mm512_mask_add_pd(f0, k0, f0, v);
      f1 = _mm512_mask_add_pd(f1, k1, f1, v);
      if (INT) {
        ip += e.icd4;
        am += e.amb;
      }
    }
    _mm512_storeu_pd(fc, f0);
    _mm512_storeu_pd(fc + 8, f1);
    ic[0] += (long)(ip & 0xffff);
    ic[5] += (long)((ip >> 16) & 0xffff);
    ic[10] += (long)((ip >> 32) & 0xffff);
    ic[15] += (long)(ip >> 48);
    ambig_count = am;
  }
#endif

  double ordered(long i, long j, const long *cells, int t, int matchMode, unsigned long *hist, long w) const
  {
    long ic[16];
    alignas(64) double fc[16];
    unsigned long ambig_count = 0;
    replay(i, j, cells, t, fc, ic, ambig_count);
    double totalNonGap = 0., nucFreq[4] = {0., 0., 0., 0.};
    for (int c1 = 0; c1 < 4; c1++) {
      for (int c2 = 0; c2 < 4; c2++) {
        double pc = (fc[c1 * 4 + c2] += (double)(ic[c1 * 4 + c2]));
        totalNonGap += pc;
        nucFreq[c1] += pc;
        nucFreq[c2] += pc;
      }
    }
    if (totalNonGap <= min_overlap) {
      return -1.;
    }
    if ((matchMode == RESOLVE || matchMode == SUBSET) && resolve_fraction < 1. &&
        totalNonGap * resolve_fraction <= ambig_count) {
      return ordered(i, j, cells, 1, AVERAGE, hist, w);
    }
    return tn93_tail_ext(nucFreq, fc[0] + fc[5] + fc[10] + fc[15], fc[2] + fc[8], fc[7] + fc[13], hist, w);
  }

  // counts for 8 lanes (pairs i x block b)
  enum { C_T, C_MM, C_TS, C_CT, C_Y1, C_Y2, C_L1, C_L2, C_T1, C_T2, C_W, C_WY, C_WL, C_WT, C_WW, NCNT };

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
  // clean-overlap counts over the K words for row i against the 8 lanes of block blk: |both clean|, and its
  // parts on columns whose base has the hi bit, the lo bit, both. Inclusion-exclusion from per-sequence
  // non-clean counts; only the row's non-clean K words are visited.
  inline void kcounts(long i, long blk, __m512i &kT, __m512i &kY, __m512i &kL, __m512i &kTT) const
  {
    const long j0 = blk * 8;
    kT = _mm512_sub_epi64(_mm512_set1_epi64(KnT - kn[0][i]), _mm512_loadu_si512(kn[0].data() + j0));
    kY = _mm512_sub_epi64(_mm512_set1_epi64(KnHi - kn[1][i]), _mm512_loadu_si512(kn[1].data() + j0));
    kL = _mm512_sub_epi64(_mm512_set1_epi64(KnLo - kn[2][i]), _mm512_loadu_si512(kn[2].data() + j0));
    kTT = _mm512_sub_epi64(_mm512_set1_epi64(KnTT - kn[3][i]), _mm512_loadu_si512(kn[3].data() + j0));
    const uint64_t *ra = P1.data() + i * WP * NP, *rb = P8.data() + blk * WP * NP * 8;
    for (uint32_t q = kd_off[i], e = kd_off[i + 1]; q < e; q++) {
      const long w = kd_list[q];
      const uint64_t nc1 = ~ra[w * NP + 2] & pvalid[w], h = KMhi[w], l = KMlo[w];
      const __m512i cm2 = _mm512_loadu_si512(rb + w * NP * 8 + 16);
      kT = _mm512_add_epi64(kT, _mm512_popcnt_epi64(_mm512_andnot_si512(cm2, _mm512_set1_epi64((long long)nc1))));
      kY = _mm512_add_epi64(kY, _mm512_popcnt_epi64(_mm512_andnot_si512(cm2, _mm512_set1_epi64((long long)(nc1 & h)))));
      kL = _mm512_add_epi64(kL, _mm512_popcnt_epi64(_mm512_andnot_si512(cm2, _mm512_set1_epi64((long long)(nc1 & l)))));
      kTT = _mm512_add_epi64(kTT,
                             _mm512_popcnt_epi64(_mm512_andnot_si512(cm2, _mm512_set1_epi64((long long)(nc1 & h & l)))));
    }
  }
#endif

  template <bool USEW, bool GM = false>
  void kernel8(long i, long blk, long out[NCNT][8]) const
  {
    const uint64_t *a = P1.data() + i * WP * NP;
    const uint64_t *b = P8.data() + blk * WP * NP * 8;
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    __m512i z = _mm512_setzero_si512();
    __m512i aT = z, aMM = z, aTS = z, aCT = z, aY1 = z, aY2 = z, aL1 = z, aL2 = z, aT1 = z, aT2 = z;
    __m512i aW = z, aWY = z, aWL = z, aWT = z, aWW = z;
    if (!USEW) {
      // V words: the pair-dependent counts (mismatches, transitions, C<->T) and the row-side base counts over
      // the clean overlap; the lane side (|both clean| and the lane's base counts) from per-sequence totals
      // minus their intersection with the row's non-clean columns
      for (long w = 0; w < WV; w++, a += NP, b += NP * 8) {
        __m512i hi1 = _mm512_set1_epi64((long long)a[0]), lo1 = _mm512_set1_epi64((long long)a[1]),
                cm1 = _mm512_set1_epi64((long long)a[2]);
        __m512i hi2 = _mm512_loadu_si512(b), lo2 = _mm512_loadu_si512(b + 8), cm2 = _mm512_loadu_si512(b + 16);
        __m512i V = _mm512_and_si512(cm1, cm2);
        __m512i x = _mm512_xor_si512(hi1, hi2), y = _mm512_xor_si512(lo1, lo2);
        __m512i mm = _mm512_ternarylogic_epi64(V, x, y, 0xE0);  // V & (x | y)
        __m512i ts = _mm512_ternarylogic_epi64(V, x, y, 0x20);  // V & ~x & y
        aMM = _mm512_add_epi64(aMM, _mm512_popcnt_epi64(mm));
        aTS = _mm512_add_epi64(aTS, _mm512_popcnt_epi64(ts));
        aCT = _mm512_add_epi64(aCT, _mm512_popcnt_epi64(_mm512_and_si512(ts, hi1)));
        aY1 = _mm512_add_epi64(aY1, _mm512_popcnt_epi64(_mm512_and_si512(hi1, V)));
        aL1 = _mm512_add_epi64(aL1, _mm512_popcnt_epi64(_mm512_and_si512(lo1, V)));
        aT1 = _mm512_add_epi64(aT1, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(hi1, lo1, V, 0x80)));
      }
      {
        const long j0 = blk * 8;
        aT = _mm512_loadu_si512(vt[0].data() + j0);
        aY2 = _mm512_loadu_si512(vt[1].data() + j0);
        aL2 = _mm512_loadu_si512(vt[2].data() + j0);
        aT2 = _mm512_loadu_si512(vt[3].data() + j0);
        const uint64_t *ra = P1.data() + i * WP * NP, *rb = P8.data() + blk * WP * NP * 8;
        for (uint32_t q = vd_off[i], e = vd_off[i + 1]; q < e; q++) {
          const long w = vd_list[q];
          const __m512i nc1 = _mm512_set1_epi64((long long)(~ra[w * NP + 2] & pvalid[w]));
          const __m512i hi2 = _mm512_loadu_si512(rb + w * NP * 8), lo2 = _mm512_loadu_si512(rb + w * NP * 8 + 8),
                        cm2 = _mm512_loadu_si512(rb + w * NP * 8 + 16);
          aT = _mm512_sub_epi64(aT, _mm512_popcnt_epi64(_mm512_and_si512(cm2, nc1)));
          aY2 = _mm512_sub_epi64(aY2, _mm512_popcnt_epi64(_mm512_and_si512(hi2, nc1)));
          aL2 = _mm512_sub_epi64(aL2, _mm512_popcnt_epi64(_mm512_and_si512(lo2, nc1)));
          aT2 = _mm512_sub_epi64(aT2, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(hi2, lo2, nc1, 0x80)));
        }
      }
    }
    for (long w = USEW ? 0 : WV; w < WP; w++, a += NP, b += NP * 8) {
      if (w >= WV && !USEW) {
        // clean-constant words: facing clean bases are equal, so no mismatches/transitions, and the base
        // counts over the clean overlap are the same for both sequences
        __m512i kT, kY, kL, kTT;
        kcounts(i, blk, kT, kY, kL, kTT);
        aT = _mm512_add_epi64(aT, kT);
        aY1 = _mm512_add_epi64(aY1, kY);
        aY2 = _mm512_add_epi64(aY2, kY);
        aL1 = _mm512_add_epi64(aL1, kL);
        aL2 = _mm512_add_epi64(aL2, kL);
        aT1 = _mm512_add_epi64(aT1, kTT);
        aT2 = _mm512_add_epi64(aT2, kTT);
        break;
      }
      __m512i hi1 = _mm512_set1_epi64((long long)a[0]), lo1 = _mm512_set1_epi64((long long)a[1]),
              cm1 = _mm512_set1_epi64((long long)a[2]);
      __m512i hi2 = _mm512_loadu_si512(b), lo2 = _mm512_loadu_si512(b + 8), cm2 = _mm512_loadu_si512(b + 16);
      __m512i V = _mm512_and_si512(cm1, cm2);
      __m512i x = _mm512_xor_si512(hi1, hi2), y = _mm512_xor_si512(lo1, lo2);
      __m512i mm = _mm512_ternarylogic_epi64(V, x, y, 0xE0);  // V & (x | y)
      __m512i ts = _mm512_ternarylogic_epi64(V, x, y, 0x20);  // V & ~x & y
      aT = _mm512_add_epi64(aT, _mm512_popcnt_epi64(V));
      aMM = _mm512_add_epi64(aMM, _mm512_popcnt_epi64(mm));
      aTS = _mm512_add_epi64(aTS, _mm512_popcnt_epi64(ts));
      aCT = _mm512_add_epi64(aCT, _mm512_popcnt_epi64(_mm512_and_si512(ts, hi1)));
      aY1 = _mm512_add_epi64(aY1, _mm512_popcnt_epi64(_mm512_and_si512(hi1, V)));
      aY2 = _mm512_add_epi64(aY2, _mm512_popcnt_epi64(_mm512_and_si512(hi2, V)));
      aL1 = _mm512_add_epi64(aL1, _mm512_popcnt_epi64(_mm512_and_si512(lo1, V)));
      aL2 = _mm512_add_epi64(aL2, _mm512_popcnt_epi64(_mm512_and_si512(lo2, V)));
      aT1 = _mm512_add_epi64(aT1, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(hi1, lo1, V, 0x80)));
      aT2 = _mm512_add_epi64(aT2, _mm512_popcnt_epi64(_mm512_ternarylogic_epi64(hi2, lo2, V, 0x80)));
      if (USEW) {
        __m512i w1 = _mm512_set1_epi64((long long)a[3]), w2 = _mm512_loadu_si512(b + 24);
        __m512i E = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, cm2), w2, cm1, 0xF8);
        __m512i hE = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, hi2), w2, hi1, 0xF8);
        __m512i lE = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, lo2), w2, lo1, 0xF8);
        aW = _mm512_add_epi64(aW, _mm512_popcnt_epi64(E));
        aWY = _mm512_add_epi64(aWY, _mm512_popcnt_epi64(hE));
        aWL = _mm512_add_epi64(aWL, _mm512_popcnt_epi64(lE));
        aWT = _mm512_add_epi64(aWT, _mm512_popcnt_epi64(_mm512_and_si512(hE, lE)));
        if (GM) {  // gap/gap sites are not compared under -a gapmm
          __m512i gg = _mm512_and_si512(_mm512_set1_epi64((long long)a[4]), _mm512_loadu_si512(b + 32));
          aWW = _mm512_add_epi64(aWW, _mm512_popcnt_epi64(_mm512_andnot_si512(gg, _mm512_and_si512(w1, w2))));
        } else
          aWW = _mm512_add_epi64(aWW, _mm512_popcnt_epi64(_mm512_and_si512(w1, w2)));
      }
    }
    _mm512_storeu_si512(out[C_T], aT);
    _mm512_storeu_si512(out[C_MM], aMM);
    _mm512_storeu_si512(out[C_TS], aTS);
    _mm512_storeu_si512(out[C_CT], aCT);
    _mm512_storeu_si512(out[C_Y1], aY1);
    _mm512_storeu_si512(out[C_Y2], aY2);
    _mm512_storeu_si512(out[C_L1], aL1);
    _mm512_storeu_si512(out[C_L2], aL2);
    _mm512_storeu_si512(out[C_T1], aT1);
    _mm512_storeu_si512(out[C_T2], aT2);
    _mm512_storeu_si512(out[C_W], aW);
    _mm512_storeu_si512(out[C_WY], aWY);
    _mm512_storeu_si512(out[C_WL], aWL);
    _mm512_storeu_si512(out[C_WT], aWT);
    _mm512_storeu_si512(out[C_WW], aWW);
#else
    for (int k = 0; k < NCNT; k++)
      for (int l = 0; l < 8; l++) out[k][l] = 0;
    for (long w = 0; w < WP; w++, a += NP, b += NP * 8)
      for (int l = 0; l < 8; l++) {
        uint64_t hi1 = a[0], lo1 = a[1], cm1 = a[2], hi2 = b[l], lo2 = b[8 + l], cm2 = b[16 + l];
        uint64_t V = cm1 & cm2, x = hi1 ^ hi2, y = lo1 ^ lo2, ts = V & ~x & y;
        out[C_T][l] += __builtin_popcountll(V);
        out[C_MM][l] += __builtin_popcountll(V & (x | y));
        out[C_TS][l] += __builtin_popcountll(ts);
        out[C_CT][l] += __builtin_popcountll(ts & hi1);
        out[C_Y1][l] += __builtin_popcountll(hi1 & V);
        out[C_Y2][l] += __builtin_popcountll(hi2 & V);
        out[C_L1][l] += __builtin_popcountll(lo1 & V);
        out[C_L2][l] += __builtin_popcountll(lo2 & V);
        out[C_T1][l] += __builtin_popcountll(hi1 & lo1 & V);
        out[C_T2][l] += __builtin_popcountll(hi2 & lo2 & V);
        if (USEW) {
          uint64_t w1 = a[3], w2 = b[24 + l];
          uint64_t E = (w1 & cm2) | (w2 & cm1), hE = (w1 & hi2) | (w2 & hi1), lE = (w1 & lo2) | (w2 & lo1);
          out[C_W][l] += __builtin_popcountll(E);
          out[C_WY][l] += __builtin_popcountll(hE);
          out[C_WL][l] += __builtin_popcountll(lE);
          out[C_WT][l] += __builtin_popcountll(hE & lE);
          out[C_WW][l] += __builtin_popcountll(w1 & w2 & ~(GM ? (a[4] & b[32 + l]) : 0ULL));
        }
      }
#endif
  }

  // N-plane counts for 8 lanes, only over the words where the row or a block lane has an N (or gap) bit
  template <bool GM>
  void kernelW(long i, long blk, long wc[NCNT][8]) const
  {
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    __m512i z = _mm512_setzero_si512(), aW = z, aWY = z, aWL = z, aWT = z, aWW = z;
    const uint64_t *ra = P1.data() + i * WP * NP, *rb = P8.data() + blk * WP * NP * 8;
    for (long m = 0; m < WM; m++) {
      uint64_t bits = rowWm[i * WM + m] | blkWm[blk * WM + m];
      while (bits) {
        long w = m * 64 + __builtin_ctzll(bits);
        bits &= bits - 1;
        const uint64_t *a = ra + w * NP, *b = rb + w * NP * 8;
        __m512i hi1 = _mm512_set1_epi64((long long)a[0]), lo1 = _mm512_set1_epi64((long long)a[1]),
                cm1 = _mm512_set1_epi64((long long)a[2]);
        __m512i hi2 = _mm512_loadu_si512(b), lo2 = _mm512_loadu_si512(b + 8), cm2 = _mm512_loadu_si512(b + 16);
        __m512i w1 = _mm512_set1_epi64((long long)a[3]), w2 = _mm512_loadu_si512(b + 24);
        __m512i E = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, cm2), w2, cm1, 0xF8);
        __m512i hE = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, hi2), w2, hi1, 0xF8);
        __m512i lE = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, lo2), w2, lo1, 0xF8);
        aW = _mm512_add_epi64(aW, _mm512_popcnt_epi64(E));
        aWY = _mm512_add_epi64(aWY, _mm512_popcnt_epi64(hE));
        aWL = _mm512_add_epi64(aWL, _mm512_popcnt_epi64(lE));
        aWT = _mm512_add_epi64(aWT, _mm512_popcnt_epi64(_mm512_and_si512(hE, lE)));
        if (GM) {
          __m512i gg = _mm512_and_si512(_mm512_set1_epi64((long long)a[4]), _mm512_loadu_si512(b + 32));
          aWW = _mm512_add_epi64(aWW, _mm512_popcnt_epi64(_mm512_andnot_si512(gg, _mm512_and_si512(w1, w2))));
        } else
          aWW = _mm512_add_epi64(aWW, _mm512_popcnt_epi64(_mm512_and_si512(w1, w2)));
      }
    }
    _mm512_storeu_si512(wc[C_W], aW);
    _mm512_storeu_si512(wc[C_WY], aWY);
    _mm512_storeu_si512(wc[C_WL], aWL);
    _mm512_storeu_si512(wc[C_WT], aWT);
    _mm512_storeu_si512(wc[C_WW], aWW);
#else
    // portable fallback: the same counts over every word
    for (int k = C_W; k <= C_WW; k++)
      for (int l = 0; l < 8; l++) wc[k][l] = 0;
    const uint64_t *ra = P1.data() + i * WP * NP, *rb = P8.data() + blk * WP * NP * 8;
    for (long w = 0; w < WP; w++)
      for (int l = 0; l < 8; l++) {
        const uint64_t *a = ra + w * NP, *b = rb + w * NP * 8;
        uint64_t h1 = a[0], l1 = a[1], m1 = a[2], h2 = b[l], l2 = b[8 + l], m2 = b[16 + l];
        uint64_t w1 = a[3], w2 = b[24 + l];
        uint64_t E = (w1 & m2) | (w2 & m1), hE = (w1 & h2) | (w2 & h1), lE = (w1 & l2) | (w2 & l1);
        wc[C_W][l] += __builtin_popcountll(E);
        wc[C_WY][l] += __builtin_popcountll(hE);
        wc[C_WL][l] += __builtin_popcountll(lE);
        wc[C_WT][l] += __builtin_popcountll(hE & lE);
        wc[C_WW][l] += __builtin_popcountll(w1 & w2 & ~(GM ? (a[4] & b[32 + l]) : 0ULL));
      }
#endif
  }

  // full 4x4 clean/clean table for 8 lanes (+ the N-plane counts into wc)
  template <bool USEW, bool GM = false>
  void kernel16(long i, long blk, long cells[16][8], long wc[NCNT][8]) const
  {
    const uint64_t *a = P1.data() + i * WP * NP;
    const uint64_t *b = P8.data() + blk * WP * NP * 8;
#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
    __m512i acc[16];
    for (int q = 0; q < 16; q++) acc[q] = _mm512_setzero_si512();
    __m512i z = _mm512_setzero_si512();
    __m512i aW = z, aWY = z, aWL = z, aWT = z, aWW = z;
    for (long w = 0; w < WP; w++, a += NP, b += NP * 8) {
      if (w >= WV && !USEW) break;  // K words: cells from kcounts below
      uint64_t h1 = a[0], l1 = a[1], m1 = a[2];
      __m512i x1[4] = {_mm512_set1_epi64((long long)(m1 & ~h1 & ~l1)), _mm512_set1_epi64((long long)(h1 & ~l1)),
                       _mm512_set1_epi64((long long)(l1 & ~h1)), _mm512_set1_epi64((long long)(h1 & l1))};
      __m512i hi2 = _mm512_loadu_si512(b), lo2 = _mm512_loadu_si512(b + 8), cm2 = _mm512_loadu_si512(b + 16);
      if (w < WV) {
        __m512i y2[4] = {_mm512_ternarylogic_epi64(cm2, hi2, lo2, 0x10), _mm512_andnot_si512(lo2, hi2),
                         _mm512_andnot_si512(hi2, lo2), _mm512_and_si512(hi2, lo2)};
        // without the N-plane counts, the A row (x = 0) is derived after the loop from the column sums
        for (int x = USEW ? 0 : 1; x < 4; x++)
          for (int y = 0; y < 4; y++)
            acc[x * 4 + y] = _mm512_add_epi64(acc[x * 4 + y], _mm512_popcnt_epi64(_mm512_and_si512(x1[x], y2[y])));
      }
      if (USEW) {
        __m512i hi1 = _mm512_set1_epi64((long long)h1), lo1 = _mm512_set1_epi64((long long)l1),
                cm1 = _mm512_set1_epi64((long long)m1);
        __m512i w1 = _mm512_set1_epi64((long long)a[3]), w2 = _mm512_loadu_si512(b + 24);
        __m512i E = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, cm2), w2, cm1, 0xF8);
        __m512i hE = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, hi2), w2, hi1, 0xF8);
        __m512i lE = _mm512_ternarylogic_epi64(_mm512_and_si512(w1, lo2), w2, lo1, 0xF8);
        aW = _mm512_add_epi64(aW, _mm512_popcnt_epi64(E));
        aWY = _mm512_add_epi64(aWY, _mm512_popcnt_epi64(hE));
        aWL = _mm512_add_epi64(aWL, _mm512_popcnt_epi64(lE));
        aWT = _mm512_add_epi64(aWT, _mm512_popcnt_epi64(_mm512_and_si512(hE, lE)));
        if (GM) {  // gap/gap sites are not compared under -a gapmm
          __m512i gg = _mm512_and_si512(_mm512_set1_epi64((long long)a[4]), _mm512_loadu_si512(b + 32));
          aWW = _mm512_add_epi64(aWW, _mm512_popcnt_epi64(_mm512_andnot_si512(gg, _mm512_and_si512(w1, w2))));
        } else
          aWW = _mm512_add_epi64(aWW, _mm512_popcnt_epi64(_mm512_and_si512(w1, w2)));
      }
    }
    if (!USEW) {
      // V words, A row: column y sums to |base y in the lane, clean in the row| = the lane's base-y total minus
      // its intersection with the row's non-clean columns; subtract the C, G, T rows
      const long j0 = blk * 8;
      const __m512i t0 = _mm512_loadu_si512(vt[0].data() + j0), t1 = _mm512_loadu_si512(vt[1].data() + j0),
                    t2 = _mm512_loadu_si512(vt[2].data() + j0), t3 = _mm512_loadu_si512(vt[3].data() + j0);
      __m512i col[4] = {_mm512_add_epi64(_mm512_sub_epi64(_mm512_sub_epi64(t0, t1), t2), t3), _mm512_sub_epi64(t1, t3),
                        _mm512_sub_epi64(t2, t3), t3};
      const uint64_t *ra = P1.data() + i * WP * NP, *rb = P8.data() + blk * WP * NP * 8;
      for (uint32_t q = vd_off[i], e = vd_off[i + 1]; q < e; q++) {
        const long w = vd_list[q];
        const __m512i nc1 = _mm512_set1_epi64((long long)(~ra[w * NP + 2] & pvalid[w]));
        const __m512i h2 = _mm512_loadu_si512(rb + w * NP * 8), l2 = _mm512_loadu_si512(rb + w * NP * 8 + 8),
                      c2 = _mm512_loadu_si512(rb + w * NP * 8 + 16);
        const __m512i yv[4] = {_mm512_ternarylogic_epi64(c2, h2, l2, 0x10), _mm512_andnot_si512(l2, h2),
                               _mm512_andnot_si512(h2, l2), _mm512_and_si512(h2, l2)};
        for (int y = 0; y < 4; y++) col[y] = _mm512_sub_epi64(col[y], _mm512_popcnt_epi64(_mm512_and_si512(yv[y], nc1)));
      }
      for (int y = 0; y < 4; y++)
        acc[y] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_sub_epi64(col[y], acc[4 + y]), acc[8 + y]), acc[12 + y]);
    }
    {
      // K words: only diagonal cells; A = T - Y - L + TT, C = Y - TT, G = L - TT, T = TT
      __m512i kT, kY, kL, kTT;
      kcounts(i, blk, kT, kY, kL, kTT);
      acc[0] = _mm512_add_epi64(acc[0], _mm512_add_epi64(_mm512_sub_epi64(_mm512_sub_epi64(kT, kY), kL), kTT));
      acc[5] = _mm512_add_epi64(acc[5], _mm512_sub_epi64(kY, kTT));
      acc[10] = _mm512_add_epi64(acc[10], _mm512_sub_epi64(kL, kTT));
      acc[15] = _mm512_add_epi64(acc[15], kTT);
    }
    for (int q = 0; q < 16; q++) _mm512_storeu_si512(cells[q], acc[q]);
    _mm512_storeu_si512(wc[C_W], aW);
    _mm512_storeu_si512(wc[C_WY], aWY);
    _mm512_storeu_si512(wc[C_WL], aWL);
    _mm512_storeu_si512(wc[C_WT], aWT);
    _mm512_storeu_si512(wc[C_WW], aWW);
#else
    for (int q = 0; q < 16; q++)
      for (int l = 0; l < 8; l++) cells[q][l] = 0;
    for (int k = C_W; k <= C_WW; k++)
      for (int l = 0; l < 8; l++) wc[k][l] = 0;
    for (long w = 0; w < WP; w++, a += NP, b += NP * 8)
      for (int l = 0; l < 8; l++) {
        uint64_t h1 = a[0], l1 = a[1], m1 = a[2], h2 = b[l], l2 = b[8 + l], m2 = b[16 + l];
        uint64_t x1[4] = {m1 & ~h1 & ~l1, h1 & ~l1, l1 & ~h1, h1 & l1};
        uint64_t y2[4] = {m2 & ~h2 & ~l2, h2 & ~l2, l2 & ~h2, h2 & l2};
        for (int x = 0; x < 4; x++)
          for (int y = 0; y < 4; y++) cells[x * 4 + y][l] += __builtin_popcountll(x1[x] & y2[y]);
        if (USEW) {
          uint64_t w1 = a[3], w2 = b[24 + l];
          uint64_t E = (w1 & m2) | (w2 & m1), hE = (w1 & h2) | (w2 & h1), lE = (w1 & l2) | (w2 & l1);
          wc[C_W][l] += __builtin_popcountll(E);
          wc[C_WY][l] += __builtin_popcountll(hE);
          wc[C_WL][l] += __builtin_popcountll(lE);
          wc[C_WT][l] += __builtin_popcountll(hE & lE);
          wc[C_WW][l] += __builtin_popcountll(w1 & w2 & ~(GM ? (a[4] & b[32 + l]) : 0ULL));
        }
      }
#endif
  }

  // Phase A for one block of 8 lanes, vectorised: aggregate the exact (x16) counts, apply tn93's -l and -g tests,
  // and hand the TN93 inputs of every fast lane to the vector tail. Lanes needing a site-order replay are flagged
  // in ordm (main mode) or with dlane = -2 (order-dependent AVERAGE fallback). Skipped pairs get dlane = -1.
  void phaseA(bool use16, const long kc[NCNT][8], const long k16[16][8], const int32_t accs[8][16], unsigned validm,
              double vnf[4][8], double vdg[8], double vag[8], double vct[8], double dlane[8], unsigned &act,
              unsigned &ordm) const
  {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    auto L = [](const long *p) { return _mm512_loadu_si512(p); };
    auto add = [](__m512i a, __m512i b) { return _mm512_add_epi64(a, b); };
    auto sub = [](__m512i a, __m512i b) { return _mm512_sub_epi64(a, b); };
    // accs[lane][slot] (int32) -> per-slot 8-lane int64 vectors: an 8x8 transpose of the slot pairs (each row
    // read as 8 int64), then sign-extension of the low / high halves (replaces 8-element gathers)
    __m512i TP[8];
    {
      __m512i r[8];
      for (int l = 0; l < 8; l++) r[l] = _mm512_loadu_si512(accs[l]);
      __m512i a0 = _mm512_unpacklo_epi64(r[0], r[1]), a1 = _mm512_unpackhi_epi64(r[0], r[1]),
              a2 = _mm512_unpacklo_epi64(r[2], r[3]), a3 = _mm512_unpackhi_epi64(r[2], r[3]),
              a4 = _mm512_unpacklo_epi64(r[4], r[5]), a5 = _mm512_unpackhi_epi64(r[4], r[5]),
              a6 = _mm512_unpacklo_epi64(r[6], r[7]), a7 = _mm512_unpackhi_epi64(r[6], r[7]);
      __m512i b0 = _mm512_shuffle_i64x2(a0, a2, 0x88), b1 = _mm512_shuffle_i64x2(a0, a2, 0xDD),
              b2 = _mm512_shuffle_i64x2(a1, a3, 0x88), b3 = _mm512_shuffle_i64x2(a1, a3, 0xDD),
              b4 = _mm512_shuffle_i64x2(a4, a6, 0x88), b5 = _mm512_shuffle_i64x2(a4, a6, 0xDD),
              b6 = _mm512_shuffle_i64x2(a5, a7, 0x88), b7 = _mm512_shuffle_i64x2(a5, a7, 0xDD);
      TP[0] = _mm512_shuffle_i64x2(b0, b4, 0x88);
      TP[4] = _mm512_shuffle_i64x2(b0, b4, 0xDD);
      TP[2] = _mm512_shuffle_i64x2(b1, b5, 0x88);
      TP[6] = _mm512_shuffle_i64x2(b1, b5, 0xDD);
      TP[1] = _mm512_shuffle_i64x2(b2, b6, 0x88);
      TP[5] = _mm512_shuffle_i64x2(b2, b6, 0xDD);
      TP[3] = _mm512_shuffle_i64x2(b3, b7, 0x88);
      TP[7] = _mm512_shuffle_i64x2(b3, b7, 0xDD);
    }
    auto accq = [&](int q) {
      const __m512i t = TP[q >> 1];
      return (q & 1) ? _mm512_srai_epi64(t, 32) : _mm512_srai_epi64(_mm512_slli_epi64(t, 32), 32);
    };
    __m512i T, D, TS, CT, nA, nC, nG;
    if (use16) {
      __m512i c[16];
      for (int q = 0; q < 16; q++) c[q] = L(k16[q]);
      T = c[0];
      for (int q = 1; q < 16; q++) T = add(T, c[q]);
      D = add(add(c[0], c[5]), add(c[10], c[15]));
      CT = add(c[7], c[13]);
      TS = add(add(c[2], c[8]), CT);
      nA = add(add(add(c[0], c[1]), add(c[2], c[3])), add(add(c[0], c[4]), add(c[8], c[12])));
      nC = add(add(add(c[4], c[5]), add(c[6], c[7])), add(add(c[1], c[5]), add(c[9], c[13])));
      nG = add(add(add(c[8], c[9]), add(c[10], c[11])), add(add(c[2], c[6]), add(c[10], c[14])));
    } else {
      T = L(kc[C_T]);
      D = sub(T, L(kc[C_MM]));
      TS = L(kc[C_TS]);
      CT = L(kc[C_CT]);
      __m512i Y = add(L(kc[C_Y1]), L(kc[C_Y2])), Lo = add(L(kc[C_L1]), L(kc[C_L2])),
              Th = add(L(kc[C_T1]), L(kc[C_T2]));
      nC = sub(Y, Th);
      nG = sub(Lo, Th);
      nA = sub(sub(sub(_mm512_slli_epi64(T, 1), nC), nG), Th);
    }
    __m512i cT = _mm512_set1_epi64(constT);
    __m512i base[7] = {_mm512_slli_epi64(add(T, cT), 4),
                       _mm512_slli_epi64(add(D, cT), 4),
                       _mm512_slli_epi64(sub(TS, CT), 4),
                       _mm512_slli_epi64(CT, 4),
                       _mm512_slli_epi64(add(nA, _mm512_set1_epi64(constNf[0])), 4),
                       _mm512_slli_epi64(add(nC, _mm512_set1_epi64(constNf[1])), 4),
                       _mm512_slli_epi64(add(nG, _mm512_set1_epi64(constNf[2])), 4)};
    __m512i wb[4], nWW = _mm512_setzero_si512();
    if (useW) {
      __m512i nW = L(kc[C_W]), nWY = L(kc[C_WY]), nWL = L(kc[C_WL]), nWT = L(kc[C_WT]);
      wb[3] = nWT;
      wb[1] = sub(nWY, nWT);
      wb[2] = sub(nWL, nWT);
      wb[0] = sub(sub(sub(nW, wb[1]), wb[2]), wb[3]);
      nWW = L(kc[C_WW]);
    }
    auto wterm = [&](int q) {
      __m512i v = _mm512_mullo_epi64(nWW, _mm512_set1_epi64(EWW.v[q]));
      for (int b = 0; b < 4; b++) v = add(v, _mm512_mullo_epi64(wb[b], _mm512_set1_epi64(EW[b].v[q])));
      return v;
    };
    __m512i ag[7];
    for (int q = 0; q < 7; q++) ag[q] = useW ? add(add(base[q], accq(q)), wterm(q)) : add(base[q], accq(q));
    __m512i flags = accq(8);
    __mmask8 vm = (__mmask8)validm;
    __mmask8 ndm = _mm512_test_epi64_mask(flags, _mm512_set1_epi64(0xffff)) & vm;
    __mmask8 nda = _mm512_test_epi64_mask(flags, _mm512_set1_epi64(0xffff0000LL));
    const __m512d sixteenth = _mm512_set1_pd(0.0625), ovl = _mm512_set1_pd((double)min_overlap);
    __m512d tng = _mm512_mul_pd(_mm512_cvtepi64_pd(ag[0]), sixteenth);
    __mmask8 live = vm & ~ndm;
    __mmask8 skip = _mm512_cmp_pd_mask(tng, ovl, _CMP_LE_OQ) & live;
    live &= ~skip;
    __mmask8 fb = 0;
    if (fallback_avg && live) {
      __m512i amb = accq(7);
      if (useW) amb = add(amb, wterm(7));
      fb = _mm512_cmp_pd_mask(_mm512_mul_pd(tng, _mm512_set1_pd(resolve_fraction)), _mm512_cvtepi64_pd(amb),
                              _CMP_LE_OQ) &
           live;
    }
    __mmask8 fbslow = fb & nda, fbfast = fb & ~nda;
    if (fbfast) {
      __m512i ag2[7];
      for (int q = 0; q < 7; q++)
        ag2[q] = useW ? add(add(base[q], accq(9 + q)), wterm(9 + q)) : add(base[q], accq(9 + q));
      __m512d tng2 = _mm512_mul_pd(_mm512_cvtepi64_pd(ag2[0]), sixteenth);
      __mmask8 skip2 = _mm512_cmp_pd_mask(tng2, ovl, _CMP_LE_OQ) & fbfast;
      skip |= skip2;
      for (int q = 0; q < 7; q++) ag[q] = _mm512_mask_blend_epi64(fbfast, ag[q], ag2[q]);
    }
    __mmask8 a = live & ~fbslow & ~skip;
    __m512d one = _mm512_set1_pd(1.0), zero = _mm512_setzero_pd();
    _mm512_storeu_pd(vnf[0], _mm512_mask_blend_pd(a, one, _mm512_mul_pd(_mm512_cvtepi64_pd(ag[4]), sixteenth)));
    _mm512_storeu_pd(vnf[1], _mm512_mask_blend_pd(a, one, _mm512_mul_pd(_mm512_cvtepi64_pd(ag[5]), sixteenth)));
    _mm512_storeu_pd(vnf[2], _mm512_mask_blend_pd(a, one, _mm512_mul_pd(_mm512_cvtepi64_pd(ag[6]), sixteenth)));
    __m512i nT = sub(sub(sub(_mm512_slli_epi64(ag[0], 1), ag[4]), ag[5]), ag[6]);
    _mm512_storeu_pd(vnf[3], _mm512_mask_blend_pd(a, one, _mm512_mul_pd(_mm512_cvtepi64_pd(nT), sixteenth)));
    _mm512_storeu_pd(vdg, _mm512_mask_blend_pd(a, zero, _mm512_mul_pd(_mm512_cvtepi64_pd(ag[1]), sixteenth)));
    _mm512_storeu_pd(vag, _mm512_mask_blend_pd(a, zero, _mm512_mul_pd(_mm512_cvtepi64_pd(ag[2]), sixteenth)));
    _mm512_storeu_pd(vct, _mm512_mask_blend_pd(a, zero, _mm512_mul_pd(_mm512_cvtepi64_pd(ag[3]), sixteenth)));
    __m512d dl = _mm512_mask_blend_pd(skip, _mm512_setzero_pd(), _mm512_set1_pd(-1.));
    dl = _mm512_mask_blend_pd(fbslow, dl, _mm512_set1_pd(-2.));
    _mm512_storeu_pd(dlane, dl);
    act = a;
    ordm = ndm;
#else
    act = 0;
    ordm = 0;
    for (int lane = 0; lane < 8; lane++) {
      vnf[0][lane] = vnf[1][lane] = vnf[2][lane] = vnf[3][lane] = 1.0;
      vdg[lane] = vag[lane] = vct[lane] = 0.0;
      dlane[lane] = 0.;
      if (!(validm >> lane & 1)) continue;
      const int32_t *acc = accs[lane];
      int nd_main = acc[8] & 0xffff, nd_avg = acc[8] >> 16;
      if (nd_main) {
        ordm |= 1u << lane;
        continue;
      }
      long T, D, TSc, CTc, nfA, nfC, nfG;
      if (use16) {
        const long c0 = k16[0][lane], c1 = k16[1][lane], c2 = k16[2][lane], c3 = k16[3][lane], c4 = k16[4][lane],
                   c5 = k16[5][lane], c6 = k16[6][lane], c7 = k16[7][lane], c8 = k16[8][lane], c9 = k16[9][lane],
                   c10 = k16[10][lane], c11 = k16[11][lane], c12 = k16[12][lane], c13 = k16[13][lane],
                   c14 = k16[14][lane], c15 = k16[15][lane];
        T = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15;
        D = c0 + c5 + c10 + c15;
        CTc = c7 + c13;
        TSc = c2 + c8 + CTc;
        nfA = c0 + c1 + c2 + c3 + c0 + c4 + c8 + c12;
        nfC = c4 + c5 + c6 + c7 + c1 + c5 + c9 + c13;
        nfG = c8 + c9 + c10 + c11 + c2 + c6 + c10 + c14;
      } else {
        T = kc[C_T][lane];
        D = T - kc[C_MM][lane];
        TSc = kc[C_TS][lane];
        CTc = kc[C_CT][lane];
        long Y = kc[C_Y1][lane] + kc[C_Y2][lane], Lo = kc[C_L1][lane] + kc[C_L2][lane],
             Th = kc[C_T1][lane] + kc[C_T2][lane];
        nfC = Y - Th;
        nfG = Lo - Th;
        nfA = 2 * T - nfC - nfG - Th;
      }
      long Tt = T + constT;
      long base[7] = {16 * Tt, 16 * (D + constT), 16 * (TSc - CTc), 16 * CTc, 16 * (nfA + constNf[0]),
                      16 * (nfC + constNf[1]), 16 * (nfG + constNf[2])};
      long wb[4] = {0, 0, 0, 0}, nWW = 0;
      if (useW) {
        long nW = kc[C_W][lane], nWY = kc[C_WY][lane], nWL = kc[C_WL][lane], nWT = kc[C_WT][lane];
        wb[3] = nWT;
        wb[1] = nWY - nWT;
        wb[2] = nWL - nWT;
        wb[0] = nW - wb[1] - wb[2] - wb[3];
        nWW = kc[C_WW][lane];
      }
      long ag[7], amb;
      for (int q = 0; q < 7; q++) {
        long v = base[q] + acc[q] + nWW * EWW.v[q];
        for (int b = 0; b < 4; b++) v += wb[b] * EW[b].v[q];
        ag[q] = v;
      }
      amb = acc[7] + nWW * EWW.v[7];
      for (int b = 0; b < 4; b++) amb += wb[b] * EW[b].v[7];
      double totalNonGap = ag[0] / 16.0;
      if (totalNonGap <= min_overlap) {
        dlane[lane] = -1.;
        continue;
      }
      if (fallback_avg && totalNonGap * resolve_fraction <= (double)(unsigned long)amb) {
        if (nd_avg) {
          dlane[lane] = -2.;
          continue;
        }
        for (int q = 0; q < 7; q++) {
          long v = base[q] + acc[9 + q] + nWW * EWW.v[9 + q];
          for (int b = 0; b < 4; b++) v += wb[b] * EW[b].v[9 + q];
          ag[q] = v;
        }
        if (ag[0] / 16.0 <= min_overlap) {
          dlane[lane] = -1.;
          continue;
        }
      }
      vnf[0][lane] = ag[4] / 16.0;
      vnf[1][lane] = ag[5] / 16.0;
      vnf[2][lane] = ag[6] / 16.0;
      vnf[3][lane] = (2 * ag[0] - ag[4] - ag[5] - ag[6]) / 16.0;
      vdg[lane] = ag[1] / 16.0;
      vag[lane] = ag[2] / 16.0;
      vct[lane] = ag[3] / 16.0;
      act |= 1u << lane;
    }
#endif
  }

  // sparse-event sums for an 8x8 tile (rows r0..r0+7, columns of block blk): out[row][lane][16]. Row-side
  // events loop once per row (8 lane accumulators), column-side events once per column (8 row accumulators),
  // so the data-dependent loop exits are paid per row/column, not per pair.
  inline void sparse_tile(long r0, long blk, int32_t out[8][8][16]) const
  {
#if defined(__AVX512F__)
    const size_t Npad = (size_t)nblocks * 8;
    const uint8_t *bcb = BC.data() + blk * 8, *bcr = BC.data() + r0;
    const Ent *ta = TA.data(), *tb = TB.data();
    for (int r = 0; r < 8; r++) {
      long i = r0 + r;
      __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
      if (i < N)
        for (uint32_t k = sp_off[i], e = sp_off[i + 1]; k < e; k++) {
          uint32_t v = sp_list[k];
          const Ent *row = ta + ((v & 31) - 4) * 18;
          const uint8_t *cs = bcb + (size_t)(v >> 5) * Npad;
          a0 = _mm512_add_epi32(a0, _mm512_loadu_si512(row[cs[0]].v));
          a1 = _mm512_add_epi32(a1, _mm512_loadu_si512(row[cs[1]].v));
          a2 = _mm512_add_epi32(a2, _mm512_loadu_si512(row[cs[2]].v));
          a3 = _mm512_add_epi32(a3, _mm512_loadu_si512(row[cs[3]].v));
          a4 = _mm512_add_epi32(a4, _mm512_loadu_si512(row[cs[4]].v));
          a5 = _mm512_add_epi32(a5, _mm512_loadu_si512(row[cs[5]].v));
          a6 = _mm512_add_epi32(a6, _mm512_loadu_si512(row[cs[6]].v));
          a7 = _mm512_add_epi32(a7, _mm512_loadu_si512(row[cs[7]].v));
        }
      _mm512_storeu_si512(out[r][0], a0);
      _mm512_storeu_si512(out[r][1], a1);
      _mm512_storeu_si512(out[r][2], a2);
      _mm512_storeu_si512(out[r][3], a3);
      _mm512_storeu_si512(out[r][4], a4);
      _mm512_storeu_si512(out[r][5], a5);
      _mm512_storeu_si512(out[r][6], a6);
      _mm512_storeu_si512(out[r][7], a7);
    }
    for (int l = 0; l < 8; l++) {
      long j = blk * 8 + l;
      if (j >= N) continue;
      uint32_t k = sp_off[j], e = sp_off[j + 1];
      if (k == e) continue;
      __m512i b0 = _mm512_setzero_si512(), b1 = b0, b2 = b0, b3 = b0, b4 = b0, b5 = b0, b6 = b0, b7 = b0;
      for (; k < e; k++) {
        uint32_t v = sp_list[k];
        const Ent *row = tb + ((v & 31) - 4) * 18;
        const uint8_t *rc = bcr + (size_t)(v >> 5) * Npad;
        b0 = _mm512_add_epi32(b0, _mm512_loadu_si512(row[rc[0]].v));
        b1 = _mm512_add_epi32(b1, _mm512_loadu_si512(row[rc[1]].v));
        b2 = _mm512_add_epi32(b2, _mm512_loadu_si512(row[rc[2]].v));
        b3 = _mm512_add_epi32(b3, _mm512_loadu_si512(row[rc[3]].v));
        b4 = _mm512_add_epi32(b4, _mm512_loadu_si512(row[rc[4]].v));
        b5 = _mm512_add_epi32(b5, _mm512_loadu_si512(row[rc[5]].v));
        b6 = _mm512_add_epi32(b6, _mm512_loadu_si512(row[rc[6]].v));
        b7 = _mm512_add_epi32(b7, _mm512_loadu_si512(row[rc[7]].v));
      }
      _mm512_storeu_si512(out[0][l], _mm512_add_epi32(_mm512_loadu_si512(out[0][l]), b0));
      _mm512_storeu_si512(out[1][l], _mm512_add_epi32(_mm512_loadu_si512(out[1][l]), b1));
      _mm512_storeu_si512(out[2][l], _mm512_add_epi32(_mm512_loadu_si512(out[2][l]), b2));
      _mm512_storeu_si512(out[3][l], _mm512_add_epi32(_mm512_loadu_si512(out[3][l]), b3));
      _mm512_storeu_si512(out[4][l], _mm512_add_epi32(_mm512_loadu_si512(out[4][l]), b4));
      _mm512_storeu_si512(out[5][l], _mm512_add_epi32(_mm512_loadu_si512(out[5][l]), b5));
      _mm512_storeu_si512(out[6][l], _mm512_add_epi32(_mm512_loadu_si512(out[6][l]), b6));
      _mm512_storeu_si512(out[7][l], _mm512_add_epi32(_mm512_loadu_si512(out[7][l]), b7));
    }
#endif
  }

  // sparse-event sums for row i against the 8 columns of block blk (lane-major accs[8][16])
  inline void sparse_block(long i, long blk, int32_t accs[8][16]) const
  {
    if (!use_bc) {
      for (int l = 0; l < 8; l++) {
        long j = blk * 8 + l;
        if (j < N)
          sparse_acc(i, j, accs[l]);
        else
          memset(accs[l], 0, 64);
      }
      return;
    }
#if defined(__AVX512F__)
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const uint8_t *bc = BC.data() + (size_t)blk * 8;
    const size_t Npad = (size_t)nblocks * 8;
    const Ent *ta = TA.data();
    for (uint32_t k = sp_off[i], e = sp_off[i + 1]; k < e; k++) {
      uint32_t v = sp_list[k];
      const Ent *row = ta + ((v & 31) - 4) * 18;
      const uint8_t *cs = bc + (size_t)(v >> 5) * Npad;
      a0 = _mm512_add_epi32(a0, _mm512_loadu_si512(row[cs[0]].v));
      a1 = _mm512_add_epi32(a1, _mm512_loadu_si512(row[cs[1]].v));
      a2 = _mm512_add_epi32(a2, _mm512_loadu_si512(row[cs[2]].v));
      a3 = _mm512_add_epi32(a3, _mm512_loadu_si512(row[cs[3]].v));
      a4 = _mm512_add_epi32(a4, _mm512_loadu_si512(row[cs[4]].v));
      a5 = _mm512_add_epi32(a5, _mm512_loadu_si512(row[cs[5]].v));
      a6 = _mm512_add_epi32(a6, _mm512_loadu_si512(row[cs[6]].v));
      a7 = _mm512_add_epi32(a7, _mm512_loadu_si512(row[cs[7]].v));
    }
    const uint8_t *ci = cc.data() + i * Lc;
    const Ent *tb = TB.data();
    const long j0 = blk * 8;
#define SWIFT_S2(L, A)                                                                    \
  if (j0 + L < N)                                                                         \
    for (uint32_t k = sp_off[j0 + L], e = sp_off[j0 + L + 1]; k < e; k++) {               \
      uint32_t v = sp_list[k];                                                            \
      A = _mm512_add_epi32(A, _mm512_loadu_si512(tb[((v & 31) - 4) * 18 + ci[v >> 5]].v)); \
    }
    SWIFT_S2(0, a0)
    SWIFT_S2(1, a1)
    SWIFT_S2(2, a2)
    SWIFT_S2(3, a3)
    SWIFT_S2(4, a4)
    SWIFT_S2(5, a5)
    SWIFT_S2(6, a6)
    SWIFT_S2(7, a7)
#undef SWIFT_S2
    _mm512_storeu_si512(accs[0], a0);
    _mm512_storeu_si512(accs[1], a1);
    _mm512_storeu_si512(accs[2], a2);
    _mm512_storeu_si512(accs[3], a3);
    _mm512_storeu_si512(accs[4], a4);
    _mm512_storeu_si512(accs[5], a5);
    _mm512_storeu_si512(accs[6], a6);
    _mm512_storeu_si512(accs[7], a7);
#else
    for (int l = 0; l < 8; l++) {
      long j = blk * 8 + l;
      if (j < N)
        sparse_acc(i, j, accs[l]);
      else
        memset(accs[l], 0, 64);
    }
#endif
  }

  // sparse-event accumulation for pair (i, j): 16 int32 lanes
  inline void sparse_acc(long i, long j, int32_t acc[16]) const
  {
    const uint8_t *c1 = cc.data() + i * Lc, *c2 = cc.data() + j * Lc;
#if defined(__AVX512F__)
    __m512i A = _mm512_setzero_si512();
    for (uint32_t k = sp_off[i], e = sp_off[i + 1]; k < e; k++) {
      uint32_t v = sp_list[k];
      const Ent &t = TA[((v & 31) - 4) * 18 + c2[v >> 5]];
      A = _mm512_add_epi32(A, _mm512_loadu_si512(t.v));
    }
    for (uint32_t k = sp_off[j], e = sp_off[j + 1]; k < e; k++) {
      uint32_t v = sp_list[k];
      const Ent &t = TB[((v & 31) - 4) * 18 + c1[v >> 5]];
      A = _mm512_add_epi32(A, _mm512_loadu_si512(t.v));
    }
    _mm512_storeu_si512(acc, A);
#else
    for (int q = 0; q < 16; q++) acc[q] = 0;
    for (uint32_t k = sp_off[i], e = sp_off[i + 1]; k < e; k++) {
      uint32_t v = sp_list[k];
      const Ent &t = TA[((v & 31) - 4) * 18 + c2[v >> 5]];
      for (int q = 0; q < 16; q++) acc[q] += t.v[q];
    }
    for (uint32_t k = sp_off[j], e = sp_off[j + 1]; k < e; k++) {
      uint32_t v = sp_list[k];
      const Ent &t = TB[((v & 31) - 4) * 18 + c1[v >> 5]];
      for (int q = 0; q < 16; q++) acc[q] += t.v[q];
    }
#endif
  }

  double slow(long i, long j, unsigned long *hist, long w) const
  {
    return ref::computeTN93(mcodes.data() + i * L, mcodes.data() + j * L, L, (char)mode, NULL, min_overlap, hist,
                            HISTOGRAM_SLICE, HISTOGRAM_BINS, w, 1L, &desc[i], &desc[j], -1.0);
  }
};

// Exact replica of the tail of tn93's computeTN93 (tn93_shared.cc lines 1016-1079) for the case where all
// float_counts are exact (dyadic): fcDiag = fc00+fc11+fc22+fc33, fcAG = fc02+fc20, fcCT = fc13+fc31.
static inline double tn93_tail_raw(const double nucFreq[4], double fcDiag, double fcAG, double fcCT)
{
  bool useK2P = false;
  long aux1;
  double auxd, fY, fR, K1 = 0, K2 = 0, K3 = 0, AG, CT, ti, tv, totalNonGap, nucF[4];

  totalNonGap = 2. / (nucFreq[0] + nucFreq[1] + nucFreq[2] + nucFreq[3]);

  auxd = 1. / (nucFreq[0] + nucFreq[1] + nucFreq[2] + nucFreq[3]);
  for (aux1 = 0; aux1 < 4; aux1++) nucF[aux1] = nucFreq[aux1] * auxd;

  fR = nucF[0] + nucF[2];
  fY = nucF[1] + nucF[3];

  if (nucFreq[0] == 0 || nucFreq[1] == 0 || nucFreq[2] == 0 || nucFreq[3] == 0) {
    useK2P = true;
  } else {
    K1 = 2. * nucF[0] * nucF[2] / fR;
    K2 = 2. * nucF[1] * nucF[3] / fY;
    K3 = 2. * (fR * fY - nucF[0] * nucF[2] * fY / fR - nucF[1] * nucF[3] * fR / fY);
  }

  AG = (fcAG)*totalNonGap;
  CT = (fcCT)*totalNonGap;
  tv = 1. - ((fcDiag)*totalNonGap + AG + CT);

  double dist;

  if (useK2P) {
    ti = AG + CT;
    AG = 1. - 2. * ti - tv;
    CT = 1. - 2. * tv;
    if (AG > 0. && CT > 0.)
      dist = -0.5 * log(AG) - 0.25 * log(CT);
    else
      dist = TN93_MAX_DIST;
  } else {
    AG = 1. - AG / K1 - 0.5 * tv / fR;
    CT = 1. - CT / K2 - 0.5 * tv / fY;
    tv = 1. - 0.5 * tv / fY / fR;
    if (AG > 0. && CT > 0. && tv > 0)
      dist = -K1 * log(AG) - K2 * log(CT) - K3 * log(tv);
    else
      dist = TN93_MAX_DIST;
  }

  return dist;
}

static inline double tn93_hist_clamp(double dist, unsigned long *histogram, long count1)
{
  if (histogram) {
    long index = floor(dist * HISTOGRAM_SLICE);
    if ((unsigned long)index >= (unsigned long)HISTOGRAM_BINS) {  // tn93 compares against an unsigned long
      index = HISTOGRAM_BINS - 1;
    }
    histogram[index] += count1 * 1L;
  }

  return dist <= 0. ? 0. : dist;
}

static inline double tn93_tail(const double nucFreq[4], double fcDiag, double fcAG, double fcCT, unsigned long *histogram,
                               long count1)
{
  return tn93_hist_clamp(tn93_tail_raw(nucFreq, fcDiag, fcAG, fcCT), histogram, count1);
}


static inline double tn93_tail_raw(const double nucFreq[4], double fcDiag, double fcAG, double fcCT);

//---------------------------------------------------------------- 8-lane formula tail

static bool g_vlog_ok = false;

// verify the vector log replica against the running libm before trusting it
static bool vlog_selftest()
{
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  double x[8], y[8];
  uint64_t st = 0x9E3779B97F4A7C15ULL;
  for (int it = 0; it < 4096; it++) {
    for (int l = 0; l < 8; l++) {
      st ^= st << 13;
      st ^= st >> 7;
      st ^= st << 17;
      double u = (double)(st >> 11) * 0x1p-53;
      int k = it & 3;
      x[l] = k == 0 ? 1.0 - u * 0x1p-6 : k == 1 ? 1.0 + u * 0x1p-5 : k == 2 ? u * 2.0 + 1e-300 : 1.0 - u * 1e-12;
      if (!(x[l] > 0)) x[l] = 0.75;
    }
    _mm512_storeu_pd(y, vlog::log8(_mm512_loadu_pd(x)));
    for (int l = 0; l < 8; l++) {
      double r = log(x[l]);
      if (memcmp(&r, &y[l], 8)) return false;
    }
  }
  return true;
#else
  return false;
#endif
}

// tn93_tail for up to 8 lanes (mask act), without the histogram update; identical IEEE operations.
// The formula tail is split in two so the main loop can issue the division-heavy first half of one row-block
// and the logs and final combination of the previous one back to back: the hardware divider (the bottleneck of
// the first half) then runs alongside other work. Same IEEE operations as tn93_tail_raw, lane by lane.
struct TailState {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  alignas(64) double as[8], cs[8], ts[8], K1[8], K2[8], K3[8];
  unsigned k2p, ok;
#else
  double nf[4][8], dg[8], ag[8], ct[8];
  unsigned act;
#endif
};

static inline void tail8a(const double nf[4][8], const double fcDiag[8], const double fcAG[8], const double fcCT[8],
                          unsigned act, TailState &S)
{
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  const __m512d one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0), half = _mm512_set1_pd(0.5),
                zero = _mm512_setzero_pd();
  __m512d n0 = _mm512_loadu_pd(nf[0]), n1 = _mm512_loadu_pd(nf[1]), n2 = _mm512_loadu_pd(nf[2]),
          n3 = _mm512_loadu_pd(nf[3]);
  __m512d sn = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(n0, n1), n2), n3);
  // inactive lanes: keep the arithmetic finite
  sn = _mm512_mask_blend_pd((__mmask8)act, one, sn);
  __m512d tng = _mm512_div_pd(two, sn);
  __m512d auxd = _mm512_div_pd(one, sn);
  __m512d f0 = _mm512_mul_pd(n0, auxd), f1 = _mm512_mul_pd(n1, auxd), f2 = _mm512_mul_pd(n2, auxd),
          f3 = _mm512_mul_pd(n3, auxd);
  __m512d fR = _mm512_add_pd(f0, f2), fY = _mm512_add_pd(f1, f3);
  __mmask8 k2p = _mm512_cmpeq_pd_mask(n0, zero) | _mm512_cmpeq_pd_mask(n1, zero) | _mm512_cmpeq_pd_mask(n2, zero) |
                 _mm512_cmpeq_pd_mask(n3, zero);
  __m512d K1 = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(two, f0), f2), fR);
  __m512d K2 = _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(two, f1), f3), fY);
  __m512d K3 = _mm512_mul_pd(
      two, _mm512_sub_pd(_mm512_sub_pd(_mm512_mul_pd(fR, fY), _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(f0, f2), fY), fR)),
                         _mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(f1, f3), fR), fY)));
  __m512d AG = _mm512_mul_pd(_mm512_loadu_pd(fcAG), tng);
  __m512d CT = _mm512_mul_pd(_mm512_loadu_pd(fcCT), tng);
  __m512d tv = _mm512_sub_pd(one, _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(_mm512_loadu_pd(fcDiag), tng), AG), CT));
  // K2P branch
  __m512d ti = _mm512_add_pd(AG, CT);
  __m512d aK = _mm512_sub_pd(_mm512_sub_pd(one, _mm512_mul_pd(two, ti)), tv);
  __m512d cK = _mm512_sub_pd(one, _mm512_mul_pd(two, tv));
  // TN93 branch
  __m512d htv = _mm512_mul_pd(half, tv);
  __m512d aT = _mm512_sub_pd(_mm512_sub_pd(one, _mm512_div_pd(AG, K1)), _mm512_div_pd(htv, fR));
  __m512d cT = _mm512_sub_pd(_mm512_sub_pd(one, _mm512_div_pd(CT, K2)), _mm512_div_pd(htv, fY));
  __m512d tT = _mm512_sub_pd(one, _mm512_div_pd(_mm512_div_pd(htv, fY), fR));
  __m512d a = _mm512_mask_blend_pd(k2p, aT, aK), c = _mm512_mask_blend_pd(k2p, cT, cK),
          t = _mm512_mask_blend_pd(k2p, tT, one);
  __mmask8 ok = _mm512_cmp_pd_mask(a, zero, _CMP_GT_OQ) & _mm512_cmp_pd_mask(c, zero, _CMP_GT_OQ) &
                _mm512_cmp_pd_mask(t, zero, _CMP_GT_OQ) & (__mmask8)act;
  __m512d la, lc, lt;
  __m512d as = _mm512_mask_blend_pd(ok, one, a), cs = _mm512_mask_blend_pd(ok, one, c),
          ts = _mm512_mask_blend_pd(ok, one, t);
  _mm512_store_pd(S.as, as);
  _mm512_store_pd(S.cs, cs);
  _mm512_store_pd(S.ts, ts);
  _mm512_store_pd(S.K1, K1);
  _mm512_store_pd(S.K2, K2);
  _mm512_store_pd(S.K3, K3);
  S.k2p = k2p;
  S.ok = ok;
#else
  memcpy(S.nf, nf, sizeof S.nf);
  memcpy(S.dg, fcDiag, sizeof S.dg);
  memcpy(S.ag, fcAG, sizeof S.ag);
  memcpy(S.ct, fcCT, sizeof S.ct);
  S.act = act;
#endif
}

static inline void tail8b(const TailState &S, double out[8])
{
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  const __m512d one = _mm512_set1_pd(1.0);
  const __m512d as = _mm512_load_pd(S.as), cs = _mm512_load_pd(S.cs), ts = _mm512_load_pd(S.ts),
                K1 = _mm512_load_pd(S.K1), K2 = _mm512_load_pd(S.K2), K3 = _mm512_load_pd(S.K3);
  const __mmask8 k2p = (__mmask8)S.k2p, ok = (__mmask8)S.ok;
  __m512d la, lc, lt;
  __mmask8 sp = vlog::special8(as) | vlog::special8(cs) | vlog::special8(ts);
  if (g_vlog_ok && !sp) {
    la = vlog::log8(as);
    lc = vlog::log8(cs);
    lt = vlog::log8(ts);
  } else {
    alignas(64) double A[8], C[8], T[8];
    _mm512_store_pd(A, as);
    _mm512_store_pd(C, cs);
    _mm512_store_pd(T, ts);
    for (int l = 0; l < 8; l++) {
      A[l] = log(A[l]);
      C[l] = log(C[l]);
      T[l] = log(T[l]);
    }
    la = _mm512_load_pd(A);
    lc = _mm512_load_pd(C);
    lt = _mm512_load_pd(T);
  }
  __m512d dK = _mm512_sub_pd(_mm512_mul_pd(_mm512_set1_pd(-0.5), la), _mm512_mul_pd(_mm512_set1_pd(0.25), lc));
  __m512d nK1 = _mm512_xor_pd(K1, _mm512_set1_pd(-0.0));  // -K1, exact
  __m512d dT = _mm512_sub_pd(_mm512_sub_pd(_mm512_mul_pd(nK1, la), _mm512_mul_pd(K2, lc)), _mm512_mul_pd(K3, lt));
  __m512d d = _mm512_mask_blend_pd(k2p, dT, dK);
  d = _mm512_mask_blend_pd(ok, _mm512_set1_pd(TN93_MAX_DIST), d);
  _mm512_storeu_pd(out, d);
#else
  for (int l = 0; l < 8; l++)
    if (S.act >> l & 1) {
      double n[4] = {S.nf[0][l], S.nf[1][l], S.nf[2][l], S.nf[3][l]};
      out[l] = tn93_tail_raw(n, S.dg[l], S.ag[l], S.ct[l]);
    }
#endif
}


static inline void tail8(const double nf[4][8], const double fcDiag[8], const double fcAG[8], const double fcCT[8],
                         unsigned act, double out[8])
{
  TailState S;
  tail8a(nf, fcDiag, fcAG, fcCT, act, S);
  tail8b(S, out);
}

//---------------------------------------------------------------- output buffers

struct OutBuf {
  char *p = nullptr;
  size_t n = 0, cap = 0;
  ~OutBuf() { free(p); }
  inline void need(size_t k)
  {
    if (n + k > cap) {
      cap = max(cap * 2, n + k + 4096);
      p = (char *)realloc(p, cap);
    }
  }
  inline void put(const char *s, size_t k)
  {
    need(k);
    memcpy(p + n, s, k);
    n += k;
  }
  inline void putc_(char c)
  {
    need(1);
    p[n++] = c;
  }
};

static void write_all(int fd, const char *p, size_t n)
{
  while (n) {
    ssize_t w = write(fd, p, n);
    if (w <= 0) {
      if (w < 0 && errno == EINTR) continue;
      return;
    }
    p += w;
    n -= (size_t)w;
  }
}

//---------------------------------------------------------------- main

int main(int argc, const char *argv[])
{
  args_t args = args_t(argc, argv);

  if (args.do_bootstrap || args.do_bootstrap_two_files || args.include_prob < 1.) {
    cerr << "swifttn93: -b, -r and -u (random resampling) are not supported" << endl;
    return 1;
  }
  if (args.hamming_skip) {
    cerr << "swifttn93: -H (Hamming early exit) is not supported" << endl;
    return 1;
  }

  ref::initAlphabets(false, args.ambigs_to_resolve, false);
  for (int c = 0; c < 256; c++) g_tab[c] = ref::validFlags[(unsigned char)toupper(c)];
  for (int c = 0; c < 128; c++) g_tab128[c] = g_tab[c] < 0 ? 0xFF : (unsigned char)g_tab[c];

  PROF(double t_start = omp_get_wtime();)
  SeqSet S;
  long firstSequenceLength = 0;
  g_scan_allowed = args.input2 == nullptr;
  if (read_fasta(args.input1, S, firstSequenceLength, args.counts_in_name) == 1) return 1;
  if (g_cs.th) {
    g_cs.done.store(true, std::memory_order_release);
    g_cs.th->join();
    delete g_cs.th;
    g_cs.th = nullptr;
  }
  unsigned long seqLengthInFile1 = S.N, seqLengthInFile2 = 0;
  if (args.input2) {
    if (read_fasta(args.input2, S, firstSequenceLength, args.counts_in_name) == 1) return 1;
    seqLengthInFile2 = S.N - seqLengthInFile1;
  }

  bool do_fst = args.input2 && args.do_fst;
  bool report_self = args.input2 == NULL && args.report_self && args.do_count == false;

  unsigned long sequenceCount = S.N,
                pairwise = (args.input2 && !do_fst) ? seqLengthInFile2 * seqLengthInFile1
                                                    : (sequenceCount - 1) * (sequenceCount) / 2;

  if (!args.quiet) {
    if (args.input2 == NULL) {
      cerr << "Read " << sequenceCount << " sequences of length " << firstSequenceLength << endl
           << "Will perform " << pairwise << " pairwise distance calculations";
    } else {
      cerr << "Read " << seqLengthInFile1 << " sequences from file 1 and " << seqLengthInFile2
           << " sequences from file 2 of length " << firstSequenceLength << endl
           << "Will perform " << pairwise << " pairwise distance calculations";
      if (do_fst) cerr << ". Running in F_ST mode";
    }
  }

  ref::resolve_fraction = args.resolve_fraction;

  int resolutionOption = RESOLVE;
  switch (args.ambig) {
    case resolve:
      resolutionOption = RESOLVE;
      break;
    case average:
      resolutionOption = AVERAGE;
      break;
    case skip:
      resolutionOption = SKIP;
      break;
    case gapmm:
      resolutionOption = GAPMM;
      break;
    case subset:
      resolutionOption = SUBSET;
      break;
  }

  PROF(double t_read = omp_get_wtime();)
  g_vlog_ok = g_cs.scanned ? g_cs.vlog_ok : vlog_selftest();
  PROF(fprintf(stderr, "PROF vlog_ok %d\n", (int)g_vlog_ok);)
  Engine E;
  E.prepare(S, firstSequenceLength, (long)seqLengthInFile1, resolutionOption, args.resolve_fraction,
            (long)args.overlap);

  PROF(double t_prep = omp_get_wtime();)
  static char obuf[1 << 16];
  setvbuf(args.output, obuf, _IOFBF, sizeof obuf);

  if (args.format != hyphy) {
    if (args.do_count == false) {
      if (args.skip_header == false) {
        if (args.format == csv)
          fprintf(args.output, "ID1%cID2%cDistance\n", args.delimiter, args.delimiter);
        else
          fprintf(args.output, "Seq1%cSeq2%cDistance\n", args.delimiter, args.delimiter);
      }
    }
  }

  // rows are written with write(2) straight from their buffers; flush what stdio holds (the header) first
  fflush(args.output);
  const int out_fd = fileno(args.output);
  const bool cross_comparison_only = (args.input2 && !do_fst);
  const long upperBound = cross_comparison_only ? (long)seqLengthInFile1 : (long)sequenceCount;
  const long N = (long)sequenceCount;
  const bool hy = args.format == hyphy;
  const bool emit = !args.do_count;

  double max_[3] = {0.0, 0.0, 0.0}, mean[3] = {0.0, 0.0, 0.0}, weighted_counts[3] = {0., 0., 0.};
  long foundLinks = 0, skipped_comparisons = 0;
  unsigned long global_hist[3][HISTOGRAM_BINS];
  memset(global_hist, 0, sizeof global_hist);
  vector<vector<pair<long, double>>> hylinks;
  if (hy && emit) hylinks.resize(N);

  const char *nm = S.names.data();
  const long *cnt = S.counts.data();
  bool unit_weights = true;  // every copy number is 1 (the phase C vector path needs unit weights)
  for (long q = 0; q < N; q++) unit_weights &= cnt[q] == 1;

  struct RowState {
    OutBuf ob;
    vector<pair<long, double>> myl;
    const char *n1 = nullptr;
    long n1L = 0, lowerBound = 0, compsSkipped = 0, local_links_found = 0, instances1 = 0;
    char pre[32];  // -f csvn: "seq1<delimiter>", formatted once per row
    int preL = 0;
    double local_max[3], local_sum[3], local_weighted[3];
  };
  constexpr long TR = 32;  // rows per tile (a multiple of 8: the sparse pass works on 8x8 sub-tiles)
  struct TileRes {
    RowState rs[TR];
  };
  // Tiles are claimed dynamically; a finished tile is parked and emitted in tile order by whichever thread
  // completes the next one, so no thread ever waits for another to finish computing.
  const long ntiles = (upperBound + TR - 1) / TR;
  long next_tile = 0, emit_next = 0;
  bool writer_active = false;  // guarded by emit_mu
  vector<TileRes *> done(ntiles, nullptr), pool;
  std::mutex emit_mu;

#pragma omp parallel
  {
    unsigned long hist[3][HISTOGRAM_BINS];
    memset(hist, 0, sizeof hist);
    // rows are processed in tiles of TR consecutive rows: each 8-column block's bit planes are loaded once
    // and reused by every row of the tile (cache blocking); rows keep their own output buffers and sums and
    // are emitted in row order
    long kc[Engine::NCNT][8];
    long k16[16][8];

    while (true) {
     const long tile = __atomic_fetch_add(&next_tile, 1L, __ATOMIC_RELAXED);
     if (tile >= ntiles) break;
     TileRes *TRes;
     {
       std::lock_guard<std::mutex> g(emit_mu);
       if (pool.empty())
         TRes = new TileRes();
       else {
         TRes = pool.back();
         pool.pop_back();
       }
     }
     RowState *rsv = TRes->rs;
     const long r0 = tile * TR, r1 = min(upperBound, r0 + TR);
     for (long seq1 = r0; seq1 < r1; seq1++) {
      RowState &Rw = rsv[seq1 - r0];
      OutBuf &ob = Rw.ob;
      ob.n = 0;
      Rw.myl.clear();
      Rw.n1 = nm + S.name_off[seq1];
      Rw.n1L = S.name_len[seq1];
      Rw.lowerBound = cross_comparison_only ? (long)seqLengthInFile1 : seq1 + 1L;
      Rw.compsSkipped = 0;
      Rw.local_links_found = 0;
      Rw.instances1 = cnt[seq1];
      if (args.format == csvn) {
        Rw.preL = fastfmt::fmt_long(Rw.pre, seq1);
        Rw.pre[Rw.preL++] = args.delimiter;
      }
      for (int q = 0; q < 3; q++) Rw.local_max[q] = Rw.local_sum[q] = Rw.local_weighted[q] = 0.;
      const char *n1 = Rw.n1;
      const long n1L = Rw.n1L, instances1 = Rw.instances1;
      double *local_weighted = Rw.local_weighted;

      if (!cross_comparison_only) {
        if (instances1 > 1) {
          unsigned long self_v_self = (instances1 * (instances1 - 1L)) >> 1;
          if (seq1 < (long)seqLengthInFile1) {
            local_weighted[0] += self_v_self;
            hist[0][0] += self_v_self;
          } else {
            local_weighted[1] += self_v_self;
            hist[1][0] += self_v_self;
          }
        }
        if (report_self) {
          if (args.format == csv) {
            ob.put(n1, n1L);
            ob.putc_(args.delimiter);
            ob.put(n1, n1L);
            ob.putc_(args.delimiter);
            ob.putc_('0');
            ob.putc_('\n');
          } else if (args.format == csvn) {
            char t[64];
            int k = fastfmt::fmt_long(t, seq1);
            t[k++] = args.delimiter;
            k += fastfmt::fmt_long(t + k, seq1);
            t[k++] = args.delimiter;
            t[k++] = '0';
            t[k++] = '\n';
            ob.put(t, k);
          }
        }
      }

     }
     auto seq2_of = [](long b, int l) { return b * 8 + l; };
     alignas(64) int32_t taccs[TR][8][16];
     for (long blk = rsv[0].lowerBound >> 3; blk * 8 < N; blk++) {
      bool tile_sparse = false;
      // -f csvn: "seq2<delimiter>" for the block's 8 columns, shared by the tile's rows
      char colstr[8][32];
      int colL[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (args.format == csvn)
        for (int l = 0; l < 8; l++) {
          colL[l] = fastfmt::fmt_long(colstr[l], blk * 8 + l);
          colstr[l][colL[l]++] = args.delimiter;
        }
#if defined(__AVX512F__)
      if (E.fast_ok && E.use_bc && (r0 & 7) == 0) {
        bool skip_all = false;
        if (E.avg_force) {  // every pair of the tile goes to the replay anyway?
          bool lanes_all = true, rows_all = true;
          for (int l = 0; l < 8; l++)
            if (blk * 8 + l < N && !E.has3[blk * 8 + l]) lanes_all = false;
          for (long r = r0; r < r1; r++)
            if (!E.has3[r]) rows_all = false;
          skip_all = lanes_all || rows_all;
        }
        if (!skip_all) {
          for (long s0 = r0; s0 < r1; s0 += 8) E.sparse_tile(s0, blk, taccs + (s0 - r0));
          tile_sparse = true;
        }
      }
#endif
      // row-blocks are finished one row later (see phase B): parked state and the phase C / finish steps
      struct Pend {
        bool valid = false;
        long seq1 = 0;
        RowState *Rw = nullptr;
        unsigned act = 0, validm = 0;
        alignas(64) double vout[8], dlane[8];
        long wlane[8], binlane[8];
        TailState ts;
      };
      Pend pends[2];
      int pslot = 0;
      auto phaseC = [&](Pend &P) {
        const long seq1 = P.seq1;
        RowState &Rw = *P.Rw;
        OutBuf &ob = Rw.ob;
        auto &myl = Rw.myl;
        const char *n1 = Rw.n1;
        const long n1L = Rw.n1L, lowerBound = Rw.lowerBound;
        long &compsSkipped = Rw.compsSkipped, &local_links_found = Rw.local_links_found;
        double *local_max = Rw.local_max, *local_sum = Rw.local_sum, *local_weighted = Rw.local_weighted;
        const unsigned act = P.act, validm = P.validm;
        const double *vout = P.vout, *dlane = P.dlane;
        const long *wlane = P.wlane, *binlane = P.binlane;
        (void)seq1;
        // phase C: in column order -- histogram, links, output, row sums
        // %g digits of reported distances computed 8 lanes at once (set below on AVX-512 IFMA builds only)
        unsigned f8m = 0;
        alignas(64) int64_t f8X[8], f8nd[8];
        alignas(64) uint64_t f8dig[8];
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        // vector path for the common block: 8 valid vector-path lanes, unit weights, one histogram and no
        // pair within the reporting range (otherwise the scalar loop below handles the block)
        bool c_done = false;
        if (act == 0xffu && validm == 0xffu && unit_weights && !do_fst) {
          const __m512d d = _mm512_load_pd(vout);
          // thisD = dist <= 0 ? 0 : dist
          const __m512d tD =
              _mm512_mask_blend_pd(_mm512_cmp_pd_mask(d, _mm512_setzero_pd(), _CMP_LE_OQ), d, _mm512_setzero_pd());
          const __mmask8 lk = _mm512_cmp_pd_mask(tD, _mm512_set1_pd(args.min_distance), _CMP_GE_OQ) &
                              _mm512_cmp_pd_mask(tD, _mm512_set1_pd(args.distance), _CMP_LE_OQ);
          if (!lk) {
            // index = floor(dist * SLICE) as a long (cvtt gives 0x8000... for NaN/out of range, as the scalar
            // conversion does), then the unsigned clamp to the last bin
            __m512i idx = _mm512_cvttpd_epi64(_mm512_roundscale_pd(_mm512_mul_pd(d, _mm512_set1_pd(HISTOGRAM_SLICE)),
                                                                   _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
            idx = _mm512_min_epu64(idx, _mm512_set1_epi64(HISTOGRAM_BINS - 1));
            alignas(64) long ix[8];
            alignas(64) double td[8];
            _mm512_store_si512(ix, idx);
            _mm512_store_pd(td, tD);
            for (int lane = 0; lane < 8; lane++) hist[0][ix[lane]] += 1;
            // row sum in column order (thisD * 1 == thisD exactly), weights, max ignoring NaN (max starts at 0)
            double s = local_sum[0];
            for (int lane = 0; lane < 8; lane++) s += td[lane];
            local_sum[0] = s;
            local_weighted[0] += 8;
            const double mx = _mm512_reduce_max_pd(
                _mm512_mask_blend_pd(_mm512_cmp_pd_mask(tD, tD, _CMP_UNORD_Q), tD, _mm512_setzero_pd()));
            if (mx > local_max[0]) local_max[0] = mx;
            c_done = true;
          }
        }
        // %g digits of the reported distances, 8 lanes at once, when a block reports several pairs
        if (!c_done && emit && act && (args.format == csv || args.format == csvn)) {
#if defined(__AVX512IFMA__) && defined(__AVX512VL__)
          const __m512d d = _mm512_load_pd(vout);
          const __m512d tD =
              _mm512_mask_blend_pd(_mm512_cmp_pd_mask(d, _mm512_setzero_pd(), _CMP_LE_OQ), d, _mm512_setzero_pd());
          const __mmask8 lk = _mm512_cmp_pd_mask(tD, _mm512_set1_pd(args.min_distance), _CMP_GE_OQ) &
                              _mm512_cmp_pd_mask(tD, _mm512_set1_pd(args.distance), _CMP_LE_OQ) & (__mmask8)act;
          if (__builtin_popcount(lk) >= 2) {
            alignas(64) double td[8];
            _mm512_store_pd(td, _mm512_maskz_mov_pd(lk, tD));  // other lanes 0: never handled by fmt_g8
            f8m = fastfmt::fmt_g8(td, f8X, f8nd, f8dig) & lk;
          }
#endif
        }
        if (!c_done)
#endif
        for (int lane = 0; lane < 8; lane++) {
          long seq2 = blk * 8 + lane;
          if (seq2 < lowerBound || seq2 >= N) continue;
          long which_bin = binlane[lane], weighted_count = wlane[lane];
          double thisD;
          if (act >> lane & 1) {
            double dist = vout[lane];
            long index = floor(dist * HISTOGRAM_SLICE);
            if ((unsigned long)index >= (unsigned long)HISTOGRAM_BINS) index = HISTOGRAM_BINS - 1;
            hist[which_bin][index] += weighted_count * 1L;
            thisD = dist <= 0. ? 0. : dist;
          } else
            thisD = dlane[lane];

          if (thisD >= args.min_distance && thisD <= args.distance) {
            local_links_found += weighted_count;
            if (emit) {
              if (args.format == csv) {
                const char *n2 = nm + S.name_off[seq2];
                long n2L = S.name_len[seq2];
                ob.need(n1L + n2L + 48);
                memcpy(ob.p + ob.n, n1, n1L);
                ob.n += n1L;
                ob.p[ob.n++] = args.delimiter;
                memcpy(ob.p + ob.n, n2, n2L);
                ob.n += n2L;
                ob.p[ob.n++] = args.delimiter;
                ob.n += (f8m >> lane & 1) ? fastfmt::fmt_g_emit(ob.p + ob.n, (int)f8X[lane], (int)f8nd[lane], f8dig[lane])
                                          : fastfmt::fmt_g(ob.p + ob.n, thisD);
                ob.p[ob.n++] = '\n';
              } else if (args.format == csvn) {
                ob.need(128);
                char *t = ob.p + ob.n;
                memcpy(t, Rw.pre, 32);  // fixed-size copies; the bytes past the lengths are overwritten next
                int k = Rw.preL;
                memcpy(t + k, colstr[lane], 32);
                k += colL[lane];
                k += (f8m >> lane & 1) ? fastfmt::fmt_g_emit(t + k, (int)f8X[lane], (int)f8nd[lane], f8dig[lane])
                                       : fastfmt::fmt_g(t + k, thisD);
                t[k++] = '\n';
                ob.n += k;
              } else {
                myl.push_back(make_pair(seq2, thisD));
              }
            }
          }

          if (thisD <= -0.5) {
            compsSkipped += 1;
          } else {
            local_sum[which_bin] += thisD * (weighted_count);
            local_weighted[which_bin] += weighted_count;
            if (thisD > local_max[which_bin]) {
              local_max[which_bin] = thisD;
            }
          }
        }
      };
      auto finish = [&](Pend &P) {
        if (P.act) tail8b(P.ts, P.vout);
        phaseC(P);
        P.valid = false;
      };
      for (long seq1 = r0; seq1 < r1; seq1++) {
        RowState &Rw = rsv[seq1 - r0];
        if (blk * 8 + 8 <= Rw.lowerBound) continue;
        OutBuf &ob = Rw.ob;
        auto &myl = Rw.myl;
        const char *n1 = Rw.n1;
        const long n1L = Rw.n1L, lowerBound = Rw.lowerBound, instances1 = Rw.instances1;
        long &compsSkipped = Rw.compsSkipped, &local_links_found = Rw.local_links_found;
        double *local_max = Rw.local_max, *local_sum = Rw.local_sum, *local_weighted = Rw.local_weighted;
        {
        PROF(unsigned long long rlast = __rdtsc();)
        alignas(64) int32_t accs_own[8][16];
        int32_t(*accs)[16] = tile_sparse ? taccs[seq1 - r0] : accs_own;
        bool use16 = false, allforced = false;
        if (E.fast_ok) {
          // -a average: a three-way code (B/D/H/V) makes a pair order-dependent, so pairs touching a sequence
          // with one go straight to the exact replay (a superset is safe: the replay is exact for any pair)
          if (E.avg_force) {
            allforced = true;
            if (!E.has3[seq1])
              for (int lane = 0; lane < 8; lane++) {
                long seq2 = blk * 8 + lane;
                if (seq2 < lowerBound || seq2 >= N) continue;
                if (!E.has3[seq2]) allforced = false;
              }
          }
          if (allforced)
            use16 = true;
          else {
            if (!tile_sparse) E.sparse_block(seq1, blk, accs);
            for (int lane = 0; lane < 8; lane++) {
              long seq2 = blk * 8 + lane;
              if (seq2 < lowerBound || seq2 >= N) continue;
              if (accs[lane][8] & 0xffff) use16 = true;  // order-dependent in the main mode
            }
          }
          PROF(TICK(0);)
          if (use16) {
            E.kernel16<false>(seq1, blk, k16, kc);
            if (E.gapmm)
              E.kernelW<true>(seq1, blk, kc);
            else if (E.useW)
              E.kernelW<false>(seq1, blk, kc);
          } else {
            E.kernel8<false>(seq1, blk, kc);
            if (E.gapmm)
              E.kernelW<true>(seq1, blk, kc);
            else if (E.useW)
              E.kernelW<false>(seq1, blk, kc);
          }
        }
        PROF(TICK(1);)
        // phase A: per-lane decisions; scalar paths finish here (they update the histogram themselves)
        alignas(64) double vnf[4][8], vdg[8], vag[8], vct[8];
        Pend &PC = pends[pslot];  // this row-block's parked state (lane decisions, tail state)
        double *dlane = PC.dlane;
        alignas(64) double ofc[16][8];
        alignas(64) long oic[16][8];
        unsigned long oamb[8];
        unsigned act = 0, ordm = 0;
        long *wlane = PC.wlane, *binlane = PC.binlane;
        unsigned validm = 0;
        for (int lane = 0; lane < 8; lane++) {
          long seq2 = blk * 8 + lane;
          if (seq2 < lowerBound || seq2 >= N) {
            wlane[lane] = 0;
            binlane[lane] = 0;
            continue;
          }
          validm |= 1u << lane;
          long which_bin = 0;
          if (do_fst) {
            if (seq1 < (long)seqLengthInFile1) {
              if (seq2 >= (long)seqLengthInFile1) which_bin = 2;
            } else
              which_bin = 1;
          }
          wlane[lane] = instances1 * cnt[seq2];
          binlane[lane] = which_bin;
        }
        if (!E.fast_ok) {
          for (int lane = 0; lane < 8; lane++) {
            if (!(validm >> lane & 1)) continue;
            PROF(__atomic_add_fetch(&prof_slow, 1, __ATOMIC_RELAXED);)
            dlane[lane] = E.slow(seq1, seq2_of(blk, lane), hist[binlane[lane]], wlane[lane]);
          }
        } else {
          if (allforced) {
            for (int lane = 0; lane < 8; lane++) {
              vnf[0][lane] = vnf[1][lane] = vnf[2][lane] = vnf[3][lane] = 1.0;
              vdg[lane] = vag[lane] = vct[lane] = 0.0;
            }
            act = 0;
            ordm = validm;
          } else
            E.phaseA(use16, kc, k16, accs, validm, vnf, vdg, vag, vct, dlane, act, ordm);
          // scalar exits: order-dependent lanes (main mode) and order-dependent -g fallbacks
          for (int lane = 0; lane < 8; lane++) {
            long seq2 = blk * 8 + lane;
            if (ordm >> lane & 1) {
              PROF(__atomic_add_fetch(&prof_slow, 1, __ATOMIC_RELAXED);)
              long cells[16];
              for (int q = 0; q < 16; q++) cells[q] = k16[q][lane];
              alignas(64) double fc[16];
              long ic[16];
              E.replay(seq1, seq2, cells, 0, fc, ic, oamb[lane]);
              for (int q = 0; q < 16; q++) {
                ofc[q][lane] = fc[q];
                oic[q][lane] = ic[q];
              }
            } else if (dlane[lane] == -2.) {
              if (!use16) {  // an order-dependent -g fallback needs the full 4x4 table
                E.kernel16<false>(seq1, blk, k16, kc);
                if (E.gapmm)
                  E.kernelW<true>(seq1, blk, kc);
                else if (E.useW)
                  E.kernelW<false>(seq1, blk, kc);
                use16 = true;
              }
              long cells[16];
              for (int q = 0; q < 16; q++) cells[q] = k16[q][lane];
              dlane[lane] = E.ordered(seq1, seq2, cells, 1, AVERAGE, hist[binlane[lane]], wlane[lane]);
            }
          }
        }
        PROF(TICK(2);)
        // phase A2: tn93's final merge of float and integer counts for the order-dependent lanes, 8 at once
        if (ordm) {
          alignas(64) double tngv[8], nfo[4][8], dgo[8], ago[8], cto[8];
#if defined(__AVX512F__) && defined(__AVX512DQ__)
          __m512d tn = _mm512_setzero_pd(), nfv[4] = {tn, tn, tn, tn}, pcv[16];
          for (int c1 = 0; c1 < 4; c1++)
            for (int c2 = 0; c2 < 4; c2++) {
              int q = c1 * 4 + c2;
              __m512d pc = _mm512_add_pd(_mm512_loadu_pd(ofc[q]), _mm512_cvtepi64_pd(_mm512_loadu_si512(oic[q])));
              pcv[q] = pc;
              tn = _mm512_add_pd(tn, pc);
              nfv[c1] = _mm512_add_pd(nfv[c1], pc);
              nfv[c2] = _mm512_add_pd(nfv[c2], pc);
            }
          _mm512_storeu_pd(tngv, tn);
          for (int k = 0; k < 4; k++) _mm512_storeu_pd(nfo[k], nfv[k]);
          _mm512_storeu_pd(dgo, _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(pcv[0], pcv[5]), pcv[10]), pcv[15]));
          _mm512_storeu_pd(ago, _mm512_add_pd(pcv[2], pcv[8]));
          _mm512_storeu_pd(cto, _mm512_add_pd(pcv[7], pcv[13]));
#else
          for (int l = 0; l < 8; l++) {
            double tn = 0., nf[4] = {0., 0., 0., 0.}, pcs[16];
            for (int c1 = 0; c1 < 4; c1++)
              for (int c2 = 0; c2 < 4; c2++) {
                double pc = ofc[c1 * 4 + c2][l] + (double)oic[c1 * 4 + c2][l];
                pcs[c1 * 4 + c2] = pc;
                tn += pc;
                nf[c1] += pc;
                nf[c2] += pc;
              }
            tngv[l] = tn;
            for (int k = 0; k < 4; k++) nfo[k][l] = nf[k];
            dgo[l] = pcs[0] + pcs[5] + pcs[10] + pcs[15];
            ago[l] = pcs[2] + pcs[8];
            cto[l] = pcs[7] + pcs[13];
          }
#endif
          for (int lane = 0; lane < 8; lane++) {
            if (!(ordm >> lane & 1)) continue;
            long seq2 = blk * 8 + lane;
            double totalNonGap = tngv[lane];
            if (totalNonGap <= E.min_overlap) {
              dlane[lane] = -1.;
            } else if ((E.mode == RESOLVE || E.mode == SUBSET) && E.resolve_fraction < 1. &&
                       totalNonGap * E.resolve_fraction <= oamb[lane]) {
              long cl[16];
              for (int q = 0; q < 16; q++) cl[q] = k16[q][lane];  // the cells the replay started from
              dlane[lane] = E.ordered(seq1, seq2, cl, 1, AVERAGE, hist[binlane[lane]], wlane[lane]);
            } else {
              for (int k = 0; k < 4; k++) vnf[k][lane] = nfo[k][lane];
              vdg[lane] = dgo[lane];
              vag[lane] = ago[lane];
              vct[lane] = cto[lane];
              act |= 1u << lane;
            }
          }
        }
        // phase B: first half of the formula (divisions) for this row-block; then the second half (logs) and
        // phase C of the previously parked row-block, so this block's divisions overlap that work
        PROF(TICK(3);)
        if (act) tail8a(vnf, vdg, vag, vct, act, PC.ts);
        if (pends[pslot ^ 1].valid) finish(pends[pslot ^ 1]);
        PC.valid = true;
        PC.seq1 = seq1;
        PC.Rw = &Rw;
        PC.act = act;
        PC.validm = validm;
        pslot ^= 1;
        PROF(TICK(4);)
        }
      }
      if (pends[pslot ^ 1].valid) finish(pends[pslot ^ 1]);  // the block's last parked row
     }

      PROF(unsigned long long ro = __rdtsc();)
      // Park the finished tile. Whoever finds the next tile in order ready and no writer active becomes the
      // writer: it takes ready tiles one at a time in order (under the lock) and writes / sums each one
      // without holding the lock, so threads parking tiles never wait for output to be written.
      bool become_writer = false;
      {
        std::lock_guard<std::mutex> g(emit_mu);
        PROF(__atomic_add_fetch(&cyc[6], __rdtsc() - ro, __ATOMIC_RELAXED);)
        done[tile] = TRes;
        if (!writer_active && emit_next < ntiles && done[emit_next]) {
          writer_active = true;
          become_writer = true;
        }
      }
      while (become_writer) {
        TileRes *ET;
        long et;
        {
          std::lock_guard<std::mutex> g(emit_mu);
          if (!(emit_next < ntiles && done[emit_next])) {
            writer_active = false;
            break;
          }
          ET = done[emit_next];
          et = emit_next;
          done[emit_next] = nullptr;
          emit_next++;
        }
        const long e0 = et * TR, e1 = min(upperBound, e0 + TR);
        for (long seq1 = e0; seq1 < e1; seq1++) {
          RowState &Rw = ET->rs[seq1 - e0];
          if (Rw.ob.n) write_all(out_fd, Rw.ob.p, Rw.ob.n);
          foundLinks += Rw.local_links_found;
          skipped_comparisons += Rw.compsSkipped;
          for (int idx = 0; idx < 3; idx++) {
            if (Rw.local_max[idx] > max_[idx]) max_[idx] = Rw.local_max[idx];
            mean[idx] += Rw.local_sum[idx];
            weighted_counts[idx] += Rw.local_weighted[idx];
          }
          for (auto &pr : Rw.myl) {
            hylinks[seq1].push_back(pr);
            hylinks[pr.first].push_back(make_pair(seq1, pr.second));
          }
        }
        {
          std::lock_guard<std::mutex> g(emit_mu);
          pool.push_back(ET);
        }
      }
    }
#pragma omp critical(hist)
    {
      for (int r = 0; r < 3; r++)
        for (int k = 0; k < HISTOGRAM_BINS; k++) global_hist[r][k] += hist[r][k];
    }
  }
  for (TileRes *t : pool) delete t;

  PROF(double t_loop = omp_get_wtime();)
  if (args.do_count == false && args.format == hyphy) {
    OutBuf ob;
    ob.putc_('{');
    char t[64];
    for (long r = 0; r < N; r++) {
      ob.put("\n{", 2);
      auto &lk = hylinks[r];
      size_t li = 0;
      for (long c = 0; c < N; c++) {
        if (c) ob.putc_(',');
        if (c == r) {
          ob.putc_('0');
        } else if (li < lk.size() && lk[li].first == c) {
          int k = fastfmt::fmt_g(t, lk[li].second);
          ob.put(t, k);
          li++;
        } else {
          ob.put("100", 3);
        }
      }
      ob.putc_('}');
      if (ob.n > (1 << 20)) {
        fwrite(ob.p, 1, ob.n, args.output);
        ob.n = 0;
      }
    }
    ob.put("\n}", 2);
    fwrite(ob.p, 1, ob.n, args.output);
  }

  PROF(fprintf(stderr, "PROF read %.3f prep %.3f loop %.3f hyphy %.3f | fast %ld slow %ld fallback %ld skipped %ld useW %d W %ld Lc %ld | Gcyc sparse %.3f kernel %.3f phaseA %.3f A2 %.3f tail %.3f phaseC %.3f orderedwait %.3f\n", t_read - t_start, t_prep - t_read, t_loop - t_prep, omp_get_wtime() - t_loop, prof_fast, prof_slow, prof_fb, prof_skip, (int)E.useW, E.W, E.Lc, cyc[0]*1e-9, cyc[1]*1e-9, cyc[2]*1e-9, cyc[3]*1e-9, cyc[4]*1e-9, cyc[5]*1e-9, cyc[6]*1e-9);)
  if (!args.quiet) {
    // tn93 prints at least one progress line, which leaves cerr at precision 3
    cerr << endl << "Progress:" << setw(8) << 100. << "% (" << setw(8) << foundLinks << " links found, "
         << setw(12) << std::setprecision(3) << (double)pairwise << " evals)";
  }

  cerr << endl;

  ostream *outStream = &cout;

  if (args.output == stdout) {
    outStream = &cerr;
  }

  (*outStream) << "{" << endl;
  (*outStream) << '\t' << "\"Actual comparisons performed\" :"

               << pairwise - skipped_comparisons << ',' << endl;
  (*outStream) << "\t\"Comparisons accounting for copy numbers \" :"
               << (weighted_counts[0] + weighted_counts[1] + weighted_counts[2]) << ',' << endl;
  (*outStream) << "\t\"Total comparisons possible\" : " << pairwise << ',' << endl;
  (*outStream) << "\t\"Links found\" : " << foundLinks << ',' << endl;
  (*outStream) << "\t\"Maximum distance\" : " << max_[0] << ',' << endl;

  if (args.input2 == NULL) {
    (*outStream) << "\t\"Sequences\" : " << sequenceCount << ',' << endl;
  } else {
    (*outStream) << "\t\"Sequences in first file\" : " << seqLengthInFile1 << ',' << endl;
    (*outStream) << "\t\"Sequences in second file\" : " << seqLengthInFile2 << ',' << endl;
  }

  if (do_fst) {
    const char *keys[4] = {"File 1", "File 2", "Between", "Combined"};
    for (unsigned long k = 0; k < 3; k++) {
      (*outStream) << "\t\"Mean distance " << keys[k] << "\" : " << mean[k] / weighted_counts[k] << ',' << endl;
    }
    double meta = (mean[0] + mean[1] + mean[2]) / (weighted_counts[0] + weighted_counts[1] + weighted_counts[2]),
           intra_pop = (mean[0] + mean[1]) / (weighted_counts[0] + weighted_counts[1]),
           pi_D = mean[2] / weighted_counts[2] - intra_pop;

    (*outStream) << "\t\"Mean distance [intra] "
                 << "\" : " << intra_pop << ',' << endl;
    (*outStream) << "\t\"Mean distance [combined]"
                 << "\" : " << meta << ',' << endl;
    (*outStream) << "\t\"F_ST\" : " << pi_D / (intra_pop + pi_D) << ',' << endl;

    for (unsigned long k = 0; k < 3; k++) {
      dump_histogram(outStream, keys[k], &(global_hist[k][0]));
      (*outStream) << ',' << endl;
    }
    for (unsigned long k = 0; k < HISTOGRAM_BINS; k++) {
      global_hist[0][k] += global_hist[1][k] + global_hist[2][k];
    }
    dump_histogram(outStream, keys[3], global_hist[0]);
  } else {
    (*outStream) << "\t\"Mean distance\" : " << mean[0] / weighted_counts[0] << ',' << endl;
    dump_histogram(outStream, NULL, global_hist[0]);
  }
  (*outStream) << '}' << endl;

  if (args.do_count) {
    fprintf(args.output, "Found %ld links among %ld pairwise comparisons\n", foundLinks,
            pairwise - skipped_comparisons);
  }

  return 0;
}
