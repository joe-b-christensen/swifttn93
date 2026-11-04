// refkernel.h -- declarations for the verbatim tn93 v1.1.0 excerpts in refkernel.cpp.
// Macros and struct copied from tn93 src/tn93_shared.h (MIT licence).
#ifndef SWIFTTN93_REFKERNEL_H
#define SWIFTTN93_REFKERNEL_H

#include <alloca.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>

#define RESOLVE 0
#define AVERAGE 1
#define SKIP 2
#define GAPMM 3
#define SUBSET 4

#define GAP 17
#define GAP_AA 24

#define IS_GAP(c, gap_val) ((c) >= (gap_val))
#define GET_JUMP(c, gap_val) ((c >= gap_val) ? (c == gap_val ? 0 : c - 1) : 0)

#define MIN(a, b) (a) < (b) ? (a) : (b)
#define MAX(a, b) (a) > (b) ? (a) : (b)

namespace ref {

struct sequence_gap_structure
{
  long first_nongap, last_nongap, resolved_start, resolved_end, total_gaps,
      total_ambigs;

  sequence_gap_structure (void)
  {
    first_nongap = LONG_MAX;
    last_nongap = 0L;
    resolved_start = 0L;
    resolved_end = 0L;
    total_gaps = 0L;
    total_ambigs = 0L;
  }
};

extern const long resolutions[][4];
extern const double resolutionsCount[];
extern unsigned char *resolveTheseAmbigs;
extern double resolve_fraction;
extern signed char validFlags[256];

void initAlphabets (bool doAminoAcid, char *resolutionSubset, bool id_map);

struct sequence_gap_structure describe_sequence (const char *source, const unsigned long sequence_length,
                                                 const unsigned long char_count, bool do_jumps);

double computeTN93 (const char *s1, const char *s2, const unsigned long L, const char matchMode,
                    const long *randomize, const long min_overlap, unsigned long *histogram,
                    const double slice, const unsigned long hist_size, const long count1, const long count2,
                    const sequence_gap_structure *sequence_descriptor1,
                    const sequence_gap_structure *sequence_descriptor2, const double threshold);

void ambiguity_handler (unsigned c1, unsigned c2, const char matchMode, double float_counts[4][4],
                        long integer_counts[4][4], unsigned long &ambig_count);

} // namespace ref

#endif
