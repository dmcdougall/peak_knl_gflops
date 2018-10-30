// This code was taken from https://colfaxresearch.com/skl-avx512
#include <stdio.h>
#include <omp.h>

const int n_trials = 1000000000; // Enough to keep cores busy for a while and observe a steady state
const int flops_per_calc = 2; // Multiply + add = 2 instructions
const int n_chained_fmas = 16; // Must be tuned for architectures here and in blocks (R) and in (E)

int main() {

#pragma omp parallel
  { // Benchmark in all threads
    double t0 = omp_get_wtime(); // start timer

    double fa[VECTOR_WIDTH*n_chained_fmas] __attribute__((aligned(64)));
    double fb[VECTOR_WIDTH] __attribute__((aligned(64)));
    double fc[VECTOR_WIDTH] __attribute__((aligned(64)));

    int i, j;

    for (i = 0; i < VECTOR_WIDTH; i++) {
      fb[i] = 0.5;
      fc[i] = 1.0;
      for (j = 0; j < n_chained_fmas; j++) {
        fa[i*n_chained_fmas+j] = 0.0;
      }
    }

    for (i = 0; i < n_trials; i++) {
      for (j = 0; j < VECTOR_WIDTH; j++) { // VECTOR_WIDTH=4 for AVX2, =8 for AVX-512
        fa[j +  0*VECTOR_WIDTH] = fa[j +  0*VECTOR_WIDTH]*fb[j] + fc[j]; // This is block (E)
        fa[j +  1*VECTOR_WIDTH] = fa[j +  1*VECTOR_WIDTH]*fb[j] + fc[j]; // To tune for a specific architecture,
        fa[j +  2*VECTOR_WIDTH] = fa[j +  2*VECTOR_WIDTH]*fb[j] + fc[j]; // more or fewer such FMA constructs
        fa[j +  3*VECTOR_WIDTH] = fa[j +  3*VECTOR_WIDTH]*fb[j] + fc[j]; // must be used
        fa[j +  4*VECTOR_WIDTH] = fa[j +  4*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j +  5*VECTOR_WIDTH] = fa[j +  5*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j +  6*VECTOR_WIDTH] = fa[j +  6*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j +  7*VECTOR_WIDTH] = fa[j +  7*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j +  8*VECTOR_WIDTH] = fa[j +  8*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j +  9*VECTOR_WIDTH] = fa[j +  9*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 10*VECTOR_WIDTH] = fa[j + 10*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 11*VECTOR_WIDTH] = fa[j + 11*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 12*VECTOR_WIDTH] = fa[j + 12*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 13*VECTOR_WIDTH] = fa[j + 13*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 14*VECTOR_WIDTH] = fa[j + 14*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 15*VECTOR_WIDTH] = fa[j + 15*VECTOR_WIDTH]*fb[j] + fc[j];
      }
    }

    double t1 = omp_get_wtime();

    // Do something with fa
    double sum = 0.0;
    for (i = 0; i < VECTOR_WIDTH*n_chained_fmas; i++) {
      sum += fa[i];
    }

#pragma omp master
    {
      const double gflops = 1.0e-9*(double)VECTOR_WIDTH*(double)n_trials*(double)flops_per_calc* (double)omp_get_max_threads()*(double)n_chained_fmas;
      printf("Sum=%.1f, Chained FMAs=%d, vector width=%d, GFLOPs=%.1f, time=%.6f s, performance=%.1f GFLOP/s\n", sum, n_chained_fmas, VECTOR_WIDTH, gflops, t1 - t0, gflops/(t1 -
t0));
    }
  }

  return 0;
}
