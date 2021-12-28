/*
 * Created by Ketan Date
 */

#ifndef _STRUCTURES_H
#define _STRUCTURES_H

struct Array
{
	int size;
	short *elements;
};

struct Matrix
{
	int rowsize;
	int colsize;
	int *elements;
};

struct Vertices
{
	short *row_assignments;
	short *col_assignments;
	int *row_covers;
	int *col_covers;	
};

struct Edges
{
	int *costs;
	char *masks;
};

struct CompactEdges
{
	short *neighbors;
	int *ptrs;
	short *is_visited;
};

struct Predicates
{
	int size;
	bool *predicates;
	int *addresses;
};

struct VertexData
{
	short *parents;
	short *children;
};

#endif