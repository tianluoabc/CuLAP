/*
 * Created by Ketan Date
 */

#ifndef _VARIABLES_H
#define _VARIABLES_H

#include "structures.h"

//#define LIGHT

#define MAX_GRIDSIZE 65535
#define INF 1000001

#define PROBLEMSIZE 5000
#define COSTRANGE 100000
#define PROBLEMCOUNT 3
#define REPETITIONS 5
#define INPUTFILE "Problemset/5000.txt"

#define SEED 24071987

#define BLOCKDIMX 16
#define BLOCKDIMY 8

#define NORMAL 'N'
#define PRIME 'P'
#define STAR 'S'
#define ZERO 'Z'

#define DORMANT 0
#define ACTIVE 1
#define VISITED 2
#define REVERSE 3
#define AUGMENT 4

extern double memory;
extern int N;
extern int N2;
extern int M;
extern Matrix h_costs, h_red_costs;
extern Vertices d_vertices;
extern Edges d_edges;
extern CompactEdges d_edges_csr, d_edges_csc;

#endif