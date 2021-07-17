/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the
        Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.

Following parameters in d_vars.h can be used to tune the performance of algorithm:

1. EPSILON: This parameter controls the tolerance on the floating point precision.
            Setting this too small will result in increased solution time because the
            algorithm will search for precise solutions. Setting it too high may cause
            some inaccuracies.

2. BLOCKDIMX, BLOCKDIMY, BLOCKDIMZ: These parameters control threads_per_block to be
            used along the given dimension. Set these according to the device
            specifications and occupancy calculation.
*/

#include <cfloat>

#ifndef VARIABLES_H_
#define VARIABLES_H_

#define MAX_GRIDSIZE 65535

#define INF FLT_MAX
#define EPSILON 0.000001

#define SEED 01010001

#define BLOCKDIMX 64
#define BLOCKDIMY 1
#define BLOCKDIMZ 1

#define DORMANT 0
#define ACTIVE 1
#define VISITED 2
#define REVERSE 3
#define AUGMENT 4
#define MODIFIED 5

#endif /* VARIABLES_H_ */
