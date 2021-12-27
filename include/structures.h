/*
 * Created by Ketan Date
 */

#ifndef _STRUCTURES_H
#define _STRUCTURES_H

struct Array
{
	int size;
	int *elements;
};

struct Matrix
{
	int rowsize;
	int colsize;
	int *elements;
};

struct Vertices
{
	int *row_assignments;
	int *col_assignments;
	int *row_covers;
	int *col_covers;	
};

struct Edges
{
	int *costs;
	int *masks;
};

struct VertexData
{
	int *parents;
	int *is_visited;
};

struct CompactEdges
{
	int *nbrs;
	int *ptrs;
};

#endif