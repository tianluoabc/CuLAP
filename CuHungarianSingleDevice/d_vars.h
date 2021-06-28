/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*
*/

#include <cfloat>

#ifndef VARIABLES_H_
#define VARIABLES_H_

#define MAX_GRIDSIZE 65535

#define INF FLT_MAX
#define EPSILON 0.000001

#define PROBLEMSIZE 1000 // Number of rows/columns
#define BATCHSIZE 10 // Number of problems in the batch
#define COSTRANGE 1000
#define PROBLEMCOUNT 1
#define REPETITIONS 1
#define DEVICECOUNT 1


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
