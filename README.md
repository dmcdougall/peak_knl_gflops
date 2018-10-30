# Observing peak performance on KNL

## Introduction

This repository contains some very simple code for observing peak performance on
Intel's Knights Landing (KNL) processors.

The files `fma.c` and `fma_24x.c` are based off code from the Colfax Research
report, [Capabilities of Intel® AVX-512 in Intel® Xeon® Scalable Processors
(Skylake)](https://colfaxresearch.com/skl-avx512).

The work here was used in a lightning talk at the [IXPUG Annual Fall Conference
2018](https://www.ixpug.org/events/ixpug-fallconf-2018) for which you can see
the
[slides](https://www.ixpug.org/components/com_solutionlibrary/assets/documents/1538587841-IXPUG_Fall_Conf_2018_paper_16%20(2)%20-%20Damon%20McDougall.pdf)
or [recorded
video](http://intelstudios.edgesuite.net/180925_IXPUG/archive/180925_IXPUG_Day1-Lightning_Talk-McDougall.html).

## Compiling and modifying

To recreate the results presented (actual peak is 6/7 of theoretical peak), use
version 18 of the Intel compiler suite.  You will also need [John McCalpin's low
overhead timers](https://github.com/jdmccalpin/low-overhead-timers) (it's just
two files):

```
$ wget https://raw.githubusercontent.com/jdmccalpin/low-overhead-timers/master/low_overhead_timers.c
$ wget https://raw.githubusercontent.com/jdmccalpin/low-overhead-timers/master/low_overhead_timers.h
$ icc -sox -O3 -xMIC-AVX512 -DVECTOR_WIDTH=8 -DUNROLLFACTOR=1 fma_24x.c low_overhead_timers.c -o fma_24x.exe
```

The `VECTOR_WIDTH` option should be 8 for AVX-512 vector registers.  If you
don't specify it then default value used is 8.

The `UNROLL_FACTOR` option tells the compiler how much to unroll the FMA loop in
the code.  If you don't specify it, the default value is 1, which pipelines 24
FMAs in this example.  A value of 2 would attempt to pipeline 48, and so on.

To generate assembly, use the `-S` flag:

```
$ icc -sox -O3 -xMIC-AVX512 -DVECTOR_WIDTH=8 -DUNROLLFACTOR=1 -S fma_24x.c -o fma_24x.s
```

Open up the assembly and look for the `vfmadd` instructions:

```
..B1.15:                        # Preds ..B1.15 ..B1.14
                                # Execution count [1.00e+06]
        addl      $1, %eax                                      #69.5 c1
        vfmadd213pd %zmm0, %zmm1, %zmm25                        #71.67 c1
        vfmadd213pd %zmm0, %zmm1, %zmm24                        #72.67 c1
        vfmadd213pd %zmm0, %zmm1, %zmm23                        #73.67 c7 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm22                        #74.67 c7
        vfmadd213pd %zmm0, %zmm1, %zmm21                        #75.67 c13 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm20                        #76.67 c13
        vfmadd213pd %zmm0, %zmm1, %zmm19                        #77.67 c19 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm18                        #78.67 c19
        vfmadd213pd %zmm0, %zmm1, %zmm17                        #79.67 c25 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm16                        #80.67 c25
        vfmadd213pd %zmm0, %zmm1, %zmm15                        #81.67 c31 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm14                        #82.67 c31
        vfmadd213pd %zmm0, %zmm1, %zmm13                        #83.67 c37 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm12                        #84.67 c37
        vfmadd213pd %zmm0, %zmm1, %zmm11                        #85.67 c43 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm10                        #86.67 c43
        vfmadd213pd %zmm0, %zmm1, %zmm9                         #87.67 c49 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm8                         #88.67 c49
        vfmadd213pd %zmm0, %zmm1, %zmm7                         #89.67 c55 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm6                         #90.67 c55
        vfmadd213pd %zmm0, %zmm1, %zmm5                         #91.67 c61 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm4                         #92.67 c61
        vfmadd213pd %zmm0, %zmm1, %zmm3                         #93.67 c67 stall 2
        vfmadd213pd %zmm0, %zmm1, %zmm2                         #94.67 c67
        cmpl      $1000000000, %eax                             #69.5 c67
        jb        ..B1.15       # Prob 99%                      #69.5 c69
                                # LOE rcx rbx r13 r14 r15 eax r12d zmm0 zmm1 zmm2 zmm3 zmm4 zmm5 zmm6 zmm7 zmm8 zmm9 zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16 zmm17 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25
```

The loop overhead here is three instructions: `addl`, `cmpl`, and `jb`.  You
can replace these with two instructions instead for lower overhead:

1.  Put `movl $1000000000, %eax` at the end of the block before `B1.15`;
2.  At the end of `B1.15`, put the new loop controls: `subl $1, %eax` and `jne ..B1.15`.

We use `subl` instead of increment or decrement because they are slow on KNL.

Putting the loop overhead at the end of the block rather than straddled around
the block offers the hardware the opportunity to issue perfect pairs of `vfma`
instructions to the allocation unit, rather than a mixed `subl`/`vmfa` pair.
