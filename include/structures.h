/*
 * Structures.h
 *
 *  Created on: Oct 30, 2014
 *      Author: ketandat
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_

struct Array
{
	long size;
	int *elements;
};

struct Matrix
{
	int rowsize;
	int colsize;
	double *elements;
};

struct Vertices
{
	int *row_assignments;
	int *col_assignments;
	int *row_covers;
	int *col_covers;
	double *row_duals;
	double *col_duals;
};


struct CompactEdges
{
	int *neighbors;
	long *ptrs;
};

struct Predicates
{
	long size;
	bool *predicates;
	long *addresses;
};

struct VertexData
{
	int *parents;
	int *children;
	int *is_visited;
};


#endif /* STRUCTURES_H_ */
