/*
 * Created by Ketan Date
 */

#include "include/hungarian_algorithm.h"
#include <cstring>
int initial_assignment_count;
int h_obj_val;
int *counts;
double *times;

// Executes Hungarian algorithm on the input cost matrix. Returns minimum cost.
int solve(int *stepcounts, double *steptimes, int &init_assignments)
{
	int step = 0;
	int total_count = 0;
	bool done = false;
	h_obj_val = 0;
	initial_assignment_count = 0;

	counts = stepcounts;
	times = steptimes;

	while (!done)
	{
		total_count++;
		switch (step)
		{
		case 0:
			counts[0]++;
			step = hungarianStep0(true);
			break;
		case 1:
			counts[1]++;
			step = hungarianStep1(true);
			break;
		case 2:
			counts[2]++;
			step = hungarianStep2(true);
			break;
		case 3:
			counts[3]++;
			step = hungarianStep3(true);
			break;
		case 4:
			///counts[4]++;
			///step = hungarianStep4(true);
			break;
		case 5:
			counts[5]++;
			step = hungarianStep5(true);
			break;
		case 6:
			counts[6]++;
			step = hungarianStep6(true);
			break;
		case 100:
			done = true;
			break;
		}
	}

	init_assignments = initial_assignment_count;

	return h_obj_val;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int hungarianStep0(bool count_time)
{
	double start = omp_get_wtime();

	rowReduction(d_edges.costs, N);	   // Kernel execution.
	columnReduction(d_edges.costs, N); // Kernel execution.

	double end = omp_get_wtime();

	if (count_time)
		times[0] += (end - start);

	return 1;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int hungarianStep1(bool count_time)
{
	double start = omp_get_wtime();

	memset(d_vertices.row_assignments, -1, N * sizeof(int));
	memset(d_vertices.col_assignments, -1, N * sizeof(int));
	memset(d_vertices.row_covers, 0, N * sizeof(int));
	memset(d_vertices.col_covers, 0, N * sizeof(int));

	computeInitialAssignments(d_edges.masks, d_edges.costs, N); // Kernel execution.

	int next = 2;

	while (true)
	{

		initial_assignment_count = 0;

		if ((next = hungarianStep2(false)) == 6)
			break;

		if ((next = hungarianStep3(false)) == 5)
			break;
	}

	double end = omp_get_wtime();

	if (count_time)
		times[1] += (end - start);

	return next;
}

// Function for checking optimality and constructing predicates and covers.
int hungarianStep2(bool count_time)
{
	int next = 3;

	double start = omp_get_wtime();

	memset(d_vertices.row_covers, 0, N * sizeof(int));
	memset(d_vertices.col_covers, 0, N * sizeof(int));

	memset(d_row_data.is_visited, DORMANT, N * sizeof(int));
	memset(d_col_data.is_visited, DORMANT, N * sizeof(int));

	int cover_count = populateAssignments(d_vertices.row_assignments, d_vertices.col_assignments, d_vertices.row_covers, d_edges.masks, d_edges.costs, N); // Kernel execution.
	initialzeVertices();

	double end = omp_get_wtime();

	if (initial_assignment_count == 0)
		initial_assignment_count = cover_count;

	if (cover_count == N)
	{

		delete[] d_edges_csr.nbrs;
		delete[] d_edges_csr.ptrs;
		d_edges_csr.nbrs = NULL;
		d_edges_csr.ptrs = NULL;

		next = 6;
	}

	if (count_time)
		times[2] += (end - start);

	return next;
}

// Function for finding minimum zero cover.
int hungarianStep3(bool count_time)
{
	int next;

	double start = omp_get_wtime();

	executeZeroCover(next); // execute zero cover algorithm.

	double end = omp_get_wtime();

	if (count_time)
		times[3] += (end - start);

	return next;
}

// Function for updating the dual variables to increase the number of uncovered zeros.
int hungarianStep5(bool count_time)
{
	int d_min = INF;

	double start = omp_get_wtime();

	dualUpdate_1(d_min, d_edges.costs, d_vertices.row_covers, d_vertices.col_covers, N);

	dualUpdate_2(d_min, d_edges.masks, d_edges.costs, d_vertices.row_covers, d_vertices.col_covers, N);

	double end = omp_get_wtime();

	if (count_time)
		times[5] += (end - start);

	return 3;
}

// Function for calculating final objective function.
int hungarianStep6(bool count_time)
{
	int d_obj_val = 0;

	double start = omp_get_wtime();

	finalCost(h_obj_val, h_costs.elements, d_vertices.row_assignments, N);

	double end = omp_get_wtime();

	if (count_time)
		times[6] += (end - start);

	return 100;
}

// Kernel for reducing the rows by subtracting row minimum from each row element.
void rowReduction(int *d_costs, int N)
{

	for (int rowid = 0; rowid < N; rowid++)
	{
		int min = INF;

		for (int colid = 0; colid < N; colid++)
		{
			int val = d_costs[rowid * N + colid];
			if (val < min)
			{
				min = val;
			}
		}

		for (int colid = 0; colid < N; colid++)
		{
			d_costs[rowid * N + colid] -= min;
		}
	}
}

// Kernel for reducing the column by subtracting column minimum from each column element.
void columnReduction(int *d_costs, int N)
{

	for (int colid = 0; colid < N; colid++)
	{
		int min = INF;

		for (int rowid = 0; rowid < N; rowid++)
		{
			int val = d_costs[rowid * N + colid];
			if (val < min)
			{
				min = val;
			}
		}

		for (int rowid = 0; rowid < N; rowid++)
		{
			d_costs[rowid * N + colid] -= min;
		}
	}
}

// Kernel for calculating initial assignments.
void computeInitialAssignments(int *d_masks, int *d_costs, int N)
{
	int *row_cover = new int[N];
	int *col_cover = new int[N];
	memset(row_cover, 0, N * sizeof(int));
	memset(col_cover, 0, N * sizeof(int));
	for (int colid = 0; colid < N; colid++)
	{
		if (col_cover[colid] == 0)
		{
			for (int rowid = 0; rowid < N; rowid++)
			{
				if (d_costs[rowid * N + colid] == 0 && row_cover[rowid] == 0)
				{
					d_vertices.row_assignments[rowid] = colid;
					d_vertices.col_assignments[colid] = rowid;
					row_cover[rowid] = 1;
					col_cover[colid] = 1;

					break;
				}
			}
		}
	}

	delete[] row_cover;
	delete[] col_cover;
}

// Kernel for populating the assignment arrays and cover arrays.
int populateAssignments(int *d_row_assignments, int *d_col_assignments, int *d_row_covers, int *d_masks, int *d_costs, int N)
{
	int cover_count = 0;

	for (int rowid = 0; rowid < N; rowid++)
	{

		if (d_row_assignments[rowid] != -1)
		{
			d_row_covers[rowid] = 1;
			cover_count++;
		}
	}

	return cover_count;
}

void initialzeVertices(void)
{
	for (int i = 0; i < N; i++)
	{
		if (d_vertices.row_covers[i] == 0)
		{

			d_row_data.is_visited[i] = ACTIVE;
		}
	}
}

// Kernel for updating the dual reduced costs in Step 5, without using atomic functions.
void dualUpdate_1(int &d_min_val, int *d_costs, int *d_row_cover, int *d_col_cover, int N)
{
	for (int colid = 0; colid < N; colid++)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			int cost = d_costs[rowid * N + colid];
			if (d_row_cover[rowid] == 0 && d_col_cover[colid] == 0)
			{
				if (cost < d_min_val)
					d_min_val = cost;
			}
		}
	}
}

// Kernel for updating the dual reduced costs in Step 5.
void dualUpdate_2(int d_min_val, int *d_masks, int *d_costs, int *d_row_cover, int *d_col_cover, int N)
{
	for (int colid = 0; colid < N; colid++)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			int id = rowid * N + colid;
			if (d_row_cover[rowid] == 0 && d_col_cover[colid] == 0)
			{
				d_costs[id] -= d_min_val;

				if (d_costs[id] == 0)
				{
					d_row_data.is_visited[rowid] = ACTIVE;
				}
			}

			else if (d_row_cover[rowid] == 1 && d_col_cover[colid] == 1)
			{
				d_costs[id] += d_min_val;
			}
		}
	}
}

// Kernel for calculating the optimal assignment cost.
void finalCost(int &d_obj_val, int *d_costs, int *d_row_assignments, int N)
{
	for (int rowid = 0; rowid < N; rowid++)
	{
		int colid = d_row_assignments[rowid];
		int id = rowid * N + colid;
		d_obj_val += d_costs[id];
	}
}