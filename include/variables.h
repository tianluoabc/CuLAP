/*
 * Created by Ketan Date
 */

#ifndef _VARIABLES_H
#define _VARIABLES_H

#include "structures.h"

#define MAX_GRIDSIZE 65535
#define INF 1000001

#define PROBLEMSIZE 10000
#define COSTRANGE 10000
#define PROBLEMCOUNT 1
#define REPETITIONS 5
#define INPUTFILE "Problemset/1000.txt"

#define SEED 24071987

#define BLOCKDIMX 16
#define BLOCKDIMY 8

#define ZERO 0
#define PRIME 1
#define STAR 2
#define NORMAL 3


#define DORMANT 0
#define ACTIVE 1
#define VISITED 2
#define REVERSE 3
#define AUGMENT 4

extern int N;
extern int N2;
extern int M;
extern Matrix h_costs;
extern Vertices d_vertices;
extern Edges d_edges;
extern VertexData d_row_data, d_col_data;
extern CompactEdges d_edges_csr;


#endif