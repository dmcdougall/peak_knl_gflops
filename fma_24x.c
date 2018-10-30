// This code was taken from https://colfaxresearch.com/skl-avx512
#include <stdio.h>
// #include <omp.h>

// unsigned long rdtsc();
// unsigned long rdtscp();
// unsigned long tacc_rdtscp(int *chip, int *core);
// int tacc_get_core_number();
// int tacc_get_socket_number();
// unsigned long rdpmc_instructions();
// unsigned long rdpmc_actual_cycles();
// unsigned long rdpmc_reference_cycles();
// unsigned long rdpmc(int c);
// float get_TSC_frequency();

#include "low_overhead_timers.h"

#ifndef VECTOR_WIDTH
#define VECTOR_WIDTH 8
#endif

#ifndef UNROLLFACTOR
#define UNROLLFACTOR 1
#endif

const int n_trials = 1000000000; // Enough to keep cores busy for a while and observe a steady state
const int flops_per_calc = 2; // Multiply + add = 2 instructions
const int n_chained_fmas = 24; // Must be tuned for architectures here and in blocks (R) and in (E)

int main() {

	float TSC_freq;
	double t0, t1;
	long tsc_start,tsc_end,ref_cycle_start,ref_cycle_end,core_cycle_start,core_cycle_end,instr_start,instr_end;
	long delta_tsc, delta_ref_cycle, delta_core_cycle, delta_instr;
	long pmc0_start, pmc1_start, pmc0_end, pmc1_end;
	long delta_pmc0, delta_pmc1;
	int pmc_width;

// #pragma omp parallel
  { // Benchmark in all threads
    // double t0 = omp_get_wtime(); // start timer

    double fa[VECTOR_WIDTH*n_chained_fmas] __attribute__((aligned(64)));
    double fb[VECTOR_WIDTH] __attribute__((aligned(64)));
    double fc[VECTOR_WIDTH] __attribute__((aligned(64)));

    int i, j;

    for (i = 0; i < VECTOR_WIDTH; i++) {
      fb[i] = 0.1;
      fc[i] = 0.2;
      for (j = 0; j < n_chained_fmas; j++) {
        fa[i*n_chained_fmas+j] = 0.0;
      }
    }

	TSC_freq = get_TSC_frequency();
	pmc_width = get_core_counter_width();

	ref_cycle_start = rdpmc_reference_cycles();
	core_cycle_start = rdpmc_actual_cycles();
	instr_start = rdpmc_instructions();
	tsc_start = rdtscp();
	pmc0_start = rdpmc(0);
	pmc1_start = rdpmc(1);

#pragma unroll_and_jam(UNROLLFACTOR)
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
        fa[j + 16*VECTOR_WIDTH] = fa[j + 16*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 17*VECTOR_WIDTH] = fa[j + 17*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 18*VECTOR_WIDTH] = fa[j + 18*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 19*VECTOR_WIDTH] = fa[j + 19*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 20*VECTOR_WIDTH] = fa[j + 20*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 21*VECTOR_WIDTH] = fa[j + 21*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 22*VECTOR_WIDTH] = fa[j + 22*VECTOR_WIDTH]*fb[j] + fc[j];
        fa[j + 23*VECTOR_WIDTH] = fa[j + 23*VECTOR_WIDTH]*fb[j] + fc[j];
      }
    }

	ref_cycle_end = rdpmc_reference_cycles();
	core_cycle_end = rdpmc_actual_cycles();
	instr_end = rdpmc_instructions();
	tsc_end = rdtscp();
	pmc0_end = rdpmc(0);
	pmc1_end = rdpmc(1);

	delta_tsc = tsc_end - tsc_start;
	delta_ref_cycle = corrected_pmc_delta(ref_cycle_end, ref_cycle_start, pmc_width);
	delta_core_cycle = corrected_pmc_delta(core_cycle_end, core_cycle_start, pmc_width);
	delta_instr = corrected_pmc_delta(instr_end, instr_start, pmc_width);
	delta_pmc0 = corrected_pmc_delta(pmc0_end, pmc0_start, pmc_width);
	delta_pmc1 = corrected_pmc_delta(pmc1_end, pmc1_start, pmc_width);

	const double utilization = (double)delta_ref_cycle / (double)delta_tsc;
	const double active_frequency = (double)delta_core_cycle / (double)delta_ref_cycle * TSC_freq;
	const double net_frequency = (double)delta_core_cycle / (double)delta_tsc * TSC_freq;
	const double instr_per_cycle = (double)delta_instr / (double)delta_core_cycle;

    // double t1 = omp_get_wtime();

    // Do something with fa
    double sum = 0.0;
    for (i = 0; i < VECTOR_WIDTH*n_chained_fmas; i++) {
      sum += fa[i];
    }

	t0 = 0.0;
	t1 = (double) (delta_tsc) / TSC_freq;

    const double gflops = 1.0e-9*(double)VECTOR_WIDTH*(double)n_trials*(double)flops_per_calc* (double)n_chained_fmas;
	printf("Sum=%.1f, Chained FMAs=%d, vector width=%d, GFLOPs=%.1f, time=%.6f s, performance=%.1f GFLOP/s\n", sum, n_chained_fmas, VECTOR_WIDTH, gflops, t1 - t0, gflops/(t1 - t0));
	printf("TSC_cycles %ld ref_cycles %ld core_cycles %ld instructions %ld pmc0 %ld pmc1 %ld\n", 
			delta_tsc, delta_ref_cycle, delta_core_cycle, delta_instr, delta_pmc0, delta_pmc1);
	printf("Utilization %f\n",utilization);
	printf("average core frequency while active %f\n",active_frequency);
	printf("average core frequency per wall second %f\n",net_frequency);
	printf("average instructions per core cycle %f\n",instr_per_cycle);
	printf("average pmc0 per core cycle %f\n",(double)delta_pmc0/(double)delta_core_cycle);
	printf("average pmc1 per core cycle %f\n",(double)delta_pmc1/(double)delta_core_cycle);

// #pragma omp master
//    {
//      const double gflops = 1.0e-9*(double)VECTOR_WIDTH*(double)n_trials*(double)flops_per_calc* (double)omp_get_max_threads()*(double)n_chained_fmas;
//      printf("Sum=%.1f, Chained FMAs=%d, vector width=%d, GFLOPs=%.1f, time=%.6f s, performance=%.1f GFLOP/s\n", sum, n_chained_fmas, VECTOR_WIDTH, gflops, t1 - t0, gflops/(t1 -
//t0));
//    }
  }

  return 0;
}
