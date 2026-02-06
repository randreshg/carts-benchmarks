#ifndef CORRELATION_H
#define CORRELATION_H

#include "../common/polybench.h"

/* Default to STANDARD_DATASET */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

#ifndef DATA_PRINTF_MODIFIER
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Define sizes if not manually specified */
#if !defined(M) && !defined(N)

#ifdef MINI_DATASET
#define M 32
#define N 32
#endif

#ifdef SMALL_DATASET
#define M 128
#define N 128
#endif

#ifdef STANDARD_DATASET
#define M 1024
#define N 1024
#endif

#ifdef LARGE_DATASET
#define M 2048
#define N 2048
#endif

#ifdef EXTRALARGE_DATASET
#define M 3096
#define N 3096
#endif

#endif /* !M && !N */

#ifndef FLOAT_N
#define FLOAT_N 3214212.01
#endif

#ifndef EPS
#define EPS 0.005
#endif

#endif /* CORRELATION_H */
