// refkernel.cpp -- VERBATIM excerpts of tn93 v1.1.0 src/tn93_shared.cc (commit 076eb0f, MIT licence,
// Sergei L. Kosakovsky Pond et al.), wrapped in namespace ref. Used (1) to build the per-site
// ambiguity contribution tables and (2) as the exact slow path for pairs whose float_counts
// depend on site order (non-dyadic ambiguity contributions). Do not edit the copied bodies.
#include <cctype>
#include "refkernel.h"
namespace ref {
#define TN93_MAX_DIST 1000.0
const char ValidChars[] = "ACGTURYSWKMBDHVN?-",
           ValidCharsAA[] = "ACDEFGHIKLMNPQRSTVWYBZX?-";

unsigned char *resolveTheseAmbigs
    = (unsigned char *)calloc (256, sizeof (unsigned char));

double resolve_fraction = 1.;

static char empty_string[] = "";

const long resolutions[][4] = {
  { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 },
  { 0, 0, 0, 1 }, { 0, 0, 0, 1 }, // U - 4
  { 1, 0, 1, 0 },                 // RESOLVE_A | RESOLVE_G, // R - 5
  { 0, 1, 0, 1 },                 // RESOLVE_C | RESOLVE_T, // Y - 6
  { 0, 1, 1, 0 },                 // RESOLVE_C | RESOLVE_G, // S - 7
  { 1, 0, 0, 1 },                 // RESOLVE_A | RESOLVE_T, // W - 8
  { 0, 0, 1, 1 },                 // RESOLVE_G | RESOLVE_T, // K - 9
  { 1, 1, 0, 0 },                 // RESOLVE_A | RESOLVE_C, // M - 10
  { 0, 1, 1, 1 }, // RESOLVE_C | RESOLVE_G | RESOLVE_T, // B - 11
  { 1, 0, 1, 1 }, // RESOLVE_A | RESOLVE_G | RESOLVE_T, // D - 12
  { 1, 1, 0, 1 }, // RESOLVE_A | RESOLVE_C | RESOLVE_T, // H - 13
  { 1, 1, 1, 0 }, // RESOLVE_A | RESOLVE_C | RESOLVE_G, // V - 14
  { 1, 1, 1, 1 }, // RESOLVE_A | RESOLVE_C | RESOLVE_G | RESOLVE_T , // N - 15
  { 1, 1, 1, 1 }, // RESOLVE_A | RESOLVE_C | RESOLVE_G | RESOLVE_T , // ? - 16
  { 0, 0, 0, 0 }  // GAP
};
#define N_CHAR 15

const double resolutionsCount[] = { 1.f,      1.f, 1.f, 1.f, 1.f,
                                    1. / 2.f, // R
                                    1. / 2.f, // Y
                                    1. / 2.f, // S
                                    1. / 2.f, // W
                                    1. / 2.f, // K
                                    1. / 2.f, // M
                                    1. / 3.f, // B
                                    1. / 3.f, // D
                                    1. / 3.f, // H
                                    1. / 3.f, // V
                                    1. / 4.f, // N
                                    1. / 4.f, // ?
                                    0.f };
signed char validFlags[256];

//---------------------------------------------------------------

void
initAlphabets (bool doAminoAcid, char *resolutionSubset, bool id_map)
{
  for (int i = 0; i < 256; i++)
    validFlags[i] = -1;

  if (doAminoAcid)
    {
      if (id_map)
        {
          for (unsigned int i = 0; i < strlen (ValidCharsAA); i++)
            validFlags[(unsigned char)ValidCharsAA[i]]
                = (unsigned char)ValidCharsAA[i];
        }
      else
        {
          for (unsigned int i = 0; i < strlen (ValidCharsAA); i++)
            validFlags[(unsigned char)ValidCharsAA[i]] = i;
        }
    }
  else
    {
      if (id_map)
        {
          for (unsigned int i = 0; i < strlen (ValidChars); i++)
            validFlags[(unsigned char)ValidChars[i]]
                = (unsigned char)ValidChars[i];
        }
      else
        {
          for (unsigned int i = 0; i < strlen (ValidChars); i++)
            validFlags[(unsigned char)ValidChars[i]] = i;
        }

      if (resolutionSubset)
        {
          unsigned long subset_length = strlen (resolutionSubset);
          for (unsigned long rc = 0; rc < subset_length; rc++)
            {
              unsigned char rcc = toupper ((resolutionSubset[rc]));
              if (validFlags[rcc] > 3)
                {
                  resolveTheseAmbigs[(unsigned char)validFlags[rcc]] = 1;
                }
            }
        }
    }
}
struct sequence_gap_structure
describe_sequence (const char *source, const unsigned long sequence_length,
                   const unsigned long char_count, bool do_jumps)
{
  sequence_gap_structure result;
  unsigned char gap_val = char_count == 4 ? GAP : GAP_AA;

  long start_run = sequence_length + 1UL, end_run = 0L;

  result.resolved_end = end_run;
  result.resolved_start = start_run;

  for (unsigned long char_idx = 0UL; char_idx < sequence_length; char_idx++)
    {
      unsigned char this_char = (unsigned char)source[char_idx];
      if (!IS_GAP (this_char, gap_val))
        {
          if (char_idx < result.first_nongap)
            {
              result.first_nongap = char_idx;
            }
          if (char_idx > result.last_nongap)
            {
              result.last_nongap = char_idx;
            }
        }
      if (this_char < char_count)
        { // not an ambig
          if (char_idx < start_run)
            {
              end_run = start_run = char_idx;
            }
          else
            {
              end_run = char_idx;
            }
        }
      else
        { // an ambig or a gap
          if (IS_GAP (this_char, gap_val))
            {
              result.total_gaps++;
            }
          else
            {
              result.total_ambigs++;
            }
          if (end_run >= start_run)
            {
              if (end_run - start_run
                  > result.resolved_end - result.resolved_start)
                {
                  result.resolved_start = start_run;
                  result.resolved_end = end_run;
                }
            }
          start_run = sequence_length + 1UL;
          end_run = 0L;
        }
    }
  if (end_run >= start_run)
    {
      if (end_run - start_run > result.resolved_end - result.resolved_start)
        {
          result.resolved_start = start_run;
          result.resolved_end = end_run;
        }
    }

  result.total_gaps -= (sequence_length - result.resolved_end - 1);

  if (do_jumps && (char_count == 4 || char_count == 20)
      && result.first_nongap < result.last_nongap)
    {
      char *mutable_source = (char *)source;
      for (unsigned long i = result.first_nongap + 1; i < result.last_nongap;)
        {
          if ((unsigned char)source[i] == gap_val)
            {
              unsigned long run_start = i;
              while (i < result.last_nongap
                     && (unsigned char)source[i] == gap_val)
                i++;
              unsigned long run_length = i - run_start;
              if (run_length > gap_val)
                {
                  if (run_length > 255)
                    {
                      mutable_source[run_start] = (unsigned char)255;
                      i = run_start + 255;
                    }
                  else
                    {
                      mutable_source[run_start] = run_length;
                    }
                }
            }
          else
            {
              i++;
            }
        }
    }

  return result;
}

double
computeTN93 (const char *__restrict__ s1, const char *__restrict__ s2,
             const unsigned long L, const char matchMode,
             const long *randomize, const long min_overlap,
             unsigned long *histogram, const double slice,
             const unsigned long hist_size, const long count1,
             const long count2,
             const sequence_gap_structure *sequence_descriptor1,
             const sequence_gap_structure *sequence_descriptor2,
             const double threshold)
{
  bool useK2P = false;
  unsigned long ambig_count = 0UL;
  long aux1;

  double auxd, nucFreq[4] = { 0., 0., 0., 0. }, fY, fR, K1, K2, K3, AG, CT, ti,
               tv, totalNonGap = 0., nucF[4],
               float_counts[4][4] = { { 0. }, { 0. }, { 0. }, { 0. } };

  long integer_counts[4][4] = { { 0L }, { 0L }, { 0L }, { 0L } },
       integer_counts2[4][4] = { { 0L }, { 0L }, { 0L }, { 0L } },
       integer_counts3[4][4] = { { 0L }, { 0L }, { 0L }, { 0L } },
       integer_counts4[4][4] = { { 0L }, { 0L }, { 0L }, { 0L } };

  bool early_exit_check = threshold > 0.0;

  const long early_exit_check_T = early_exit_check ? (long)(threshold * L) : L;

  auto check_early_exit = [&] (long TL) -> bool {
    long differences = (integer_counts[0][1] + integer_counts2[0][1]
                        + integer_counts3[0][1] + integer_counts4[0][1])
                       + (integer_counts[0][2] + integer_counts2[0][2]
                          + integer_counts3[0][2] + integer_counts4[0][2])
                       + (integer_counts[0][3] + integer_counts2[0][3]
                          + integer_counts3[0][3] + integer_counts4[0][3])
                       + (integer_counts[1][0] + integer_counts2[1][0]
                          + integer_counts3[1][0] + integer_counts4[1][0])
                       + (integer_counts[1][2] + integer_counts2[1][2]
                          + integer_counts3[1][2] + integer_counts4[1][2])
                       + (integer_counts[1][3] + integer_counts2[1][3]
                          + integer_counts3[1][3] + integer_counts4[1][3])
                       + (integer_counts[2][0] + integer_counts2[2][0]
                          + integer_counts3[2][0] + integer_counts4[2][0])
                       + (integer_counts[2][1] + integer_counts2[2][1]
                          + integer_counts3[2][1] + integer_counts4[2][1])
                       + (integer_counts[2][3] + integer_counts2[2][3]
                          + integer_counts3[2][3] + integer_counts4[2][3])
                       + (integer_counts[3][0] + integer_counts2[3][0]
                          + integer_counts3[3][0] + integer_counts4[3][0])
                       + (integer_counts[3][1] + integer_counts2[3][1]
                          + integer_counts3[3][1] + integer_counts4[3][1])
                       + (integer_counts[3][2] + integer_counts2[3][2]
                          + integer_counts3[3][2] + integer_counts4[3][2]);
    return differences > TL;
  };

  auto ambiguityHandler = [&] (unsigned c1, unsigned c2) -> void {
    if (c1 < 4UL)
      { // c1 resolved and c2 is not
        if (matchMode != SKIP)
          {
            if (resolutionsCount[c2] > 0.)
              {
                if (matchMode == RESOLVE
                    || (matchMode == SUBSET && resolveTheseAmbigs[c2]))
                  {
                    if (resolutions[c2][c1])
                      {
                        ambig_count++;
                        integer_counts[c1][c1]++;
                        return;
                      }
                  }

                if (resolutions[c2][0])
                  float_counts[c1][0] += resolutionsCount[c2];
                if (resolutions[c2][1])
                  float_counts[c1][1] += resolutionsCount[c2];
                if (resolutions[c2][2])
                  float_counts[c1][2] += resolutionsCount[c2];
                if (resolutions[c2][3])
                  float_counts[c1][3] += resolutionsCount[c2];
              }
          }
      }
    else
      {
        if (matchMode != SKIP)
          {
            if (c2 < 4UL)
              { // c2 resolved an c1 is not
                if (resolutionsCount[c1] > 0.)
                  {
                    if (matchMode == RESOLVE
                        || (matchMode == SUBSET && resolveTheseAmbigs[c1]))
                      {
                        if (resolutions[c1][c2])
                          {
                            ambig_count++;
                            integer_counts[c2][c2]++;
                            return;
                          }
                      }

                    if (resolutions[c1][0])
                      float_counts[0][c2] += resolutionsCount[c1];
                    if (resolutions[c1][1])
                      float_counts[1][c2] += resolutionsCount[c1];
                    if (resolutions[c1][2])
                      float_counts[2][c2] += resolutionsCount[c1];
                    if (resolutions[c1][3])
                      float_counts[3][c2] += resolutionsCount[c1];
                  }
              }
            else
              {
                // ambig and ambig
                double norm = resolutionsCount[c1] * resolutionsCount[c2];
                // cout << int(c1) << ":" << int(c2) << "/" << norm << endl;
                if (norm > 0.0)
                  {
                    if (matchMode == RESOLVE
                        || (matchMode == SUBSET && resolveTheseAmbigs[c1]
                            && resolveTheseAmbigs[c2]))
                      {
                        ambig_count++;
                        long matched_count = 0L,
                             positive_match[4] = { 0, 0, 0, 0 };
                        for (long i = 0; i < 4L; i++)
                          {
                            if (resolutions[c1][i] && resolutions[c2][i])
                              {
                                matched_count++;
                                positive_match[i] = 1;
                              }
                          }

                        if (matched_count > 0L)
                          {
                            double norm2 = 1. / matched_count;

                            for (long i = 0; i < 4L; i++)
                              {
                                if (positive_match[i])
                                  {
                                    float_counts[i][i] += norm2;
                                  }
                              }
                            return;
                          }
                      }

                    for (long i = 0; i < 4L; i++)
                      {
                        if (resolutions[c1][i])
                          {
                            for (long j = 0; j < 4L; j++)
                              {
                                if (resolutions[c2][j])
                                  {
                                    float_counts[i][j] += norm;
                                  }
                              }
                          }
                      }
                  }
              }
          }
      }
  };

  if (randomize == NULL)
    {
      if (sequence_descriptor1 && sequence_descriptor2 && matchMode != GAPMM)
        {
          // #pragma omp critical
          //       cout << "HERE" << endl;

          unsigned long first_nongap
              = MAX (sequence_descriptor1->first_nongap,
                     sequence_descriptor2->first_nongap),
              last_nongap = MIN (sequence_descriptor1->last_nongap,
                                 sequence_descriptor2->last_nongap),
              span_start = MAX (sequence_descriptor1->resolved_start,
                                sequence_descriptor2->resolved_start),
              span_end = MIN (sequence_descriptor1->resolved_end,
                              sequence_descriptor2->resolved_end);
          // #pragma omp critical
          //       cout << first_nongap << " " << last_nongap << " " <<
          //       span_start << " " << span_end << endl;

          if (span_start > span_end)
            {
              for (unsigned long p = first_nongap; p <= last_nongap; p++)
                {
                  unsigned char c1 = (unsigned char)s1[p],
                                c2 = (unsigned char)s2[p];

                  if (__builtin_expect ((c1 | c2) < 4, 1))
                    {
                      integer_counts[c1][c2]++;
                    }
                  else
                    { // not both resolved
                      if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                        {
                          p += MAX (GET_JUMP (c1, GAP), GET_JUMP (c2, GAP));
                          continue;
                        }
                      ambiguityHandler (c1, c2);
                    }
                }
            }
          else
            {
#define WORTH_DOING 1024
#define PERIODIC_CHECK 128
              unsigned long p = first_nongap;

              int ac_count = 0;
              int *ac_pairs
                  = (int *)alloca ((sequence_descriptor1->total_ambigs
                                    + sequence_descriptor2->total_ambigs)
                                   * sizeof (int));

              if (span_start + WORTH_DOING > span_end)
                {

                  while (p + PERIODIC_CHECK <= last_nongap)
                    {
                      unsigned long p2 = p;
                      for (; p2 <= p + PERIODIC_CHECK; p2++)
                        {
                          unsigned char c1 = (unsigned char)s1[p2],
                                        c2 = (unsigned char)s2[p2];

                          if (__builtin_expect ((c1 | c2) < 4, 1))
                            {
                              integer_counts[c1][c2]++;
                            }
                          else
                            { // not both resolved
                              if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                                {
                                  p2 += MAX (GET_JUMP (c1, GAP),
                                             GET_JUMP (c2, GAP));
                                }
                              else
                                {
                                  ac_pairs[ac_count++] = (c1 << 8) + c2;
                                }
                            }
                        }
                      p = p2;
                      if (early_exit_check
                          && check_early_exit (early_exit_check_T))
                        {
                          return 1.0;
                        }
                    }

                  for (; p <= last_nongap; p++)
                    {
                      unsigned char c1 = (unsigned char)s1[p],
                                    c2 = (unsigned char)s2[p];

                      if (__builtin_expect ((c1 | c2) < 4, 1))
                        {
                          integer_counts[c1][c2]++;
                        }
                      else
                        { // not both resolved
                          if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                            {
                              p += MAX (GET_JUMP (c1, GAP),
                                        GET_JUMP (c2, GAP));
                            }
                          else
                            {
                              ac_pairs[ac_count++] = (c1 << 8) + c2;
                            }
                        }
                    }
                }
              else
                {
                  while (p < span_start)
                    {
                      unsigned char c1 = (unsigned char)s1[p],
                                    c2 = (unsigned char)s2[p];

                      if (__builtin_expect ((c1 | c2) < 4, 1))
                        {
                          integer_counts[c1][c2]++;
                          p++;
                        }
                      else
                        { // not both resolved
                          if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                            {
                              p += MAX (GET_JUMP (c1, GAP), GET_JUMP (c2, GAP))
                                   + 1;
                            }
                          else
                            {
                              ac_pairs[ac_count++] = (c1 << 8) + c2;
                              p++;
                            }
                        }
                    }

                  if (early_exit_check
                      && check_early_exit (early_exit_check_T))
                    {
                      return 1.0;
                    }

                  if (p < span_start)
                    p = span_start;

                  if (p <= span_end)
                    {
                      if (threshold > 0.0)
                        {
                          while (p + PERIODIC_CHECK <= span_end)
                            {
                              for (unsigned long block_end
                                   = p + PERIODIC_CHECK;
                                   p < block_end; p += 4)
                                {
                                  integer_counts[(unsigned char)s1[p]]
                                                [(unsigned char)s2[p]]++;
                                  integer_counts2[(unsigned char)s1[p + 1]]
                                                 [(unsigned char)s2[p + 1]]++;
                                  integer_counts3[(unsigned char)s1[p + 2]]
                                                 [(unsigned char)s2[p + 2]]++;
                                  integer_counts4[(unsigned char)s1[p + 3]]
                                                 [(unsigned char)s2[p + 3]]++;
                                }
                              if (early_exit_check
                                  && check_early_exit (early_exit_check_T))
                                {
                                  return 1.0;
                                }
                            }
                        }

                      for (; p + 4 <= span_end; p += 4)
                        {
                          integer_counts[(unsigned char)s1[p]]
                                        [(unsigned char)s2[p]]++;
                          integer_counts2[(unsigned char)s1[p + 1]]
                                         [(unsigned char)s2[p + 1]]++;
                          integer_counts3[(unsigned char)s1[p + 2]]
                                         [(unsigned char)s2[p + 2]]++;
                          integer_counts4[(unsigned char)s1[p + 3]]
                                         [(unsigned char)s2[p + 3]]++;
                        }
                      for (; p <= span_end; p++)
                        {
                          integer_counts[(unsigned char)s1[p]]
                                        [(unsigned char)s2[p]]++;
                        }

                      if (early_exit_check
                          && check_early_exit (early_exit_check_T))
                        {
                          return 1.0;
                        }
                    }

                  if (p < span_end + 1UL)
                    p = span_end + 1UL;

                  for (; p <= last_nongap; p++)
                    {
                      unsigned char c1 = (unsigned char)s1[p],
                                    c2 = (unsigned char)s2[p];

                      if (__builtin_expect ((c1 | c2) < 4, 1))
                        {
                          integer_counts[c1][c2]++;
                        }
                      else
                        { // not both resolved
                          if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                            {
                              p += MAX (GET_JUMP (c1, GAP),
                                        GET_JUMP (c2, GAP));
                            }
                          else
                            {
                              ac_pairs[ac_count++] = (c1 << 8) + c2;
                            }
                        }
                    }
                }
              for (int i = 0; i < ac_count; i++)
                {
                  int ac = ac_pairs[i];
                  ambiguityHandler (ac >> 8, ac & 255);
                }

              if (early_exit_check && check_early_exit (early_exit_check_T))
                {
                  return 1.0;
                }
            }
        }
      else
        {
          for (unsigned long p = 0; p < L; p++)
            {
              unsigned char c1 = (unsigned char)s1[p],
                            c2 = (unsigned char)s2[p];

              if (__builtin_expect ((c1 | c2) < 4, 1))
                {
                  integer_counts[c1][c2]++;
                }
              else
                { // not both resolved
                  if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                    {
                      if (matchMode != GAPMM)
                        {
                          p += MAX (GET_JUMP (c1, GAP), GET_JUMP (c2, GAP));
                        }
                      else
                        {
                          if (IS_GAP (c1, GAP) && IS_GAP (c2, GAP))
                            {
                              // skip
                            }
                          else
                            {
                              unsigned char tc1 = c1, tc2 = c2;
                              if (IS_GAP (c1, GAP))
                                {
                                  tc1 = N_CHAR;
                                }
                              else
                                {
                                  tc2 = N_CHAR;
                                }
                              ambiguityHandler (tc1, tc2);
                            }
                          continue;
                        }
                    }

                  ambiguityHandler (c1, c2);
                }
            }
        }
    }
  else
    {
      for (unsigned long p = 0; p < L; p++)
        {
          long pi = randomize[p];
          unsigned char c1 = (unsigned char)s1[pi], c2 = (unsigned char)s2[pi];

          if (__builtin_expect (c1 < 4 && c2 < 4, 1))
            {
              integer_counts[c1][c2]++;
            }
          else
            { // not both resolved
              if (IS_GAP (c1, GAP) || IS_GAP (c2, GAP))
                {
                  if (matchMode != GAPMM)
                    {
                      continue;
                    }
                  else
                    {
                      if (IS_GAP (c1, GAP) && IS_GAP (c2, GAP))
                        continue;
                      else
                        {
                          unsigned char tc1 = c1, tc2 = c2;
                          if (IS_GAP (c1, GAP))
                            {
                              tc1 = N_CHAR;
                            }
                          else
                            {
                              tc2 = N_CHAR;
                            }
                          ambiguityHandler (tc1, tc2);
                        }
                    }
                }
              else
                {
                  ambiguityHandler (c1, c2);
                }
            }
        }
    }

  // printf ("\n");
  for (int c1 = 0; c1 < 4; c1++)
    {
      // printf ("\n");
      for (int c2 = 0; c2 < 4; c2++)
        {
          double pc
              = (float_counts[c1][c2]
                 += (double)(integer_counts[c1][c2] + integer_counts2[c1][c2]
                             + integer_counts3[c1][c2]
                             + integer_counts4[c1][c2]));
          // printf ("%12.2g\t", pc);
          totalNonGap += pc;
          nucFreq[c1] += pc;
          nucFreq[c2] += pc;
        }
    }
  // printf ("\ntotalNonGap = %g\n", totalNonGap);

  if (totalNonGap <= min_overlap)
    {
      return -1.;
    }

  if (early_exit_check && check_early_exit (totalNonGap * threshold))
    {
      return 1.0;
    }

  if ((matchMode == RESOLVE || matchMode == SUBSET) && resolve_fraction < 1.
      && totalNonGap * resolve_fraction <= ambig_count)
    {
      // cout << ambig_count << "/" << totalNonGap << endl;
      return computeTN93 (s1, s2, L, AVERAGE, randomize, min_overlap,
                          histogram, slice, hist_size, count1, count2,
                          sequence_descriptor1, sequence_descriptor2,
                          threshold);
    }

  totalNonGap = 2. / (nucFreq[0] + nucFreq[1] + nucFreq[2] + nucFreq[3]);

  auxd = 1. / (nucFreq[0] + nucFreq[1] + nucFreq[2] + nucFreq[3]);
  for (aux1 = 0; aux1 < 4; aux1++)
    nucF[aux1] = nucFreq[aux1] * auxd;

  fR = nucF[0] + nucF[2];
  fY = nucF[1] + nucF[3];

  if (nucFreq[0] == 0 || nucFreq[1] == 0 || nucFreq[2] == 0 || nucFreq[3] == 0)
    {
      useK2P = true;
    }
  else
    {
      K1 = 2. * nucF[0] * nucF[2] / fR;
      K2 = 2. * nucF[1] * nucF[3] / fY;
      K3 = 2.
           * (fR * fY - nucF[0] * nucF[2] * fY / fR
              - nucF[1] * nucF[3] * fR / fY);
    }

  AG = (float_counts[0][2] + float_counts[2][0]) * totalNonGap;
  CT = (float_counts[1][3] + float_counts[3][1]) * totalNonGap;
  tv = 1.
       - ((float_counts[0][0] + float_counts[1][1] + float_counts[2][2]
           + float_counts[3][3])
              * totalNonGap
          + AG + CT);

  double dist;

  if (useK2P)
    {
      ti = AG + CT;
      AG = 1. - 2. * ti - tv;
      CT = 1. - 2. * tv;
      if (AG > 0. && CT > 0.)
        dist = -0.5 * log (AG) - 0.25 * log (CT);
      else
        dist = TN93_MAX_DIST;
    }
  else
    {
      AG = 1. - AG / K1 - 0.5 * tv / fR;
      CT = 1. - CT / K2 - 0.5 * tv / fY;
      tv = 1. - 0.5 * tv / fY / fR;
      if (AG > 0. && CT > 0. && tv > 0)
        dist = -K1 * log (AG) - K2 * log (CT) - K3 * log (tv);
      else
        dist = TN93_MAX_DIST;
    }

  if (histogram)
    {
      long index = floor (dist * slice);
      if (index >= hist_size)
        {
          index = hist_size - 1;
        }
      histogram[index] += count1 * count2;
    }

  return dist <= 0. ? 0. : dist; // this is to avoid returning -0
}

// The ambiguityHandler lambda of computeTN93 (lines 531-644), copied verbatim as a function.
void ambiguity_handler (unsigned c1, unsigned c2, const char matchMode, double float_counts[4][4],
                        long integer_counts[4][4], unsigned long &ambig_count)
{
  auto ambiguityHandler = [&] (unsigned c1, unsigned c2) -> void {
    if (c1 < 4UL)
      { // c1 resolved and c2 is not
        if (matchMode != SKIP)
          {
            if (resolutionsCount[c2] > 0.)
              {
                if (matchMode == RESOLVE
                    || (matchMode == SUBSET && resolveTheseAmbigs[c2]))
                  {
                    if (resolutions[c2][c1])
                      {
                        ambig_count++;
                        integer_counts[c1][c1]++;
                        return;
                      }
                  }

                if (resolutions[c2][0])
                  float_counts[c1][0] += resolutionsCount[c2];
                if (resolutions[c2][1])
                  float_counts[c1][1] += resolutionsCount[c2];
                if (resolutions[c2][2])
                  float_counts[c1][2] += resolutionsCount[c2];
                if (resolutions[c2][3])
                  float_counts[c1][3] += resolutionsCount[c2];
              }
          }
      }
    else
      {
        if (matchMode != SKIP)
          {
            if (c2 < 4UL)
              { // c2 resolved an c1 is not
                if (resolutionsCount[c1] > 0.)
                  {
                    if (matchMode == RESOLVE
                        || (matchMode == SUBSET && resolveTheseAmbigs[c1]))
                      {
                        if (resolutions[c1][c2])
                          {
                            ambig_count++;
                            integer_counts[c2][c2]++;
                            return;
                          }
                      }

                    if (resolutions[c1][0])
                      float_counts[0][c2] += resolutionsCount[c1];
                    if (resolutions[c1][1])
                      float_counts[1][c2] += resolutionsCount[c1];
                    if (resolutions[c1][2])
                      float_counts[2][c2] += resolutionsCount[c1];
                    if (resolutions[c1][3])
                      float_counts[3][c2] += resolutionsCount[c1];
                  }
              }
            else
              {
                // ambig and ambig
                double norm = resolutionsCount[c1] * resolutionsCount[c2];
                // cout << int(c1) << ":" << int(c2) << "/" << norm << endl;
                if (norm > 0.0)
                  {
                    if (matchMode == RESOLVE
                        || (matchMode == SUBSET && resolveTheseAmbigs[c1]
                            && resolveTheseAmbigs[c2]))
                      {
                        ambig_count++;
                        long matched_count = 0L,
                             positive_match[4] = { 0, 0, 0, 0 };
                        for (long i = 0; i < 4L; i++)
                          {
                            if (resolutions[c1][i] && resolutions[c2][i])
                              {
                                matched_count++;
                                positive_match[i] = 1;
                              }
                          }

                        if (matched_count > 0L)
                          {
                            double norm2 = 1. / matched_count;

                            for (long i = 0; i < 4L; i++)
                              {
                                if (positive_match[i])
                                  {
                                    float_counts[i][i] += norm2;
                                  }
                              }
                            return;
                          }
                      }

                    for (long i = 0; i < 4L; i++)
                      {
                        if (resolutions[c1][i])
                          {
                            for (long j = 0; j < 4L; j++)
                              {
                                if (resolutions[c2][j])
                                  {
                                    float_counts[i][j] += norm;
                                  }
                              }
                          }
                      }
                  }
              }
          }
      }
  };
  ambiguityHandler (c1, c2);
}
} // namespace ref
