/*
 * SqLAP.cpp
 *
 *  Created on: Jun 15, 2015
 *      Author: date2
 */

#include "include/sqlap.h"

SqLAP::SqLAP(int _size)
{

	N = _size;
	N2 = N * N;
	M = 0;

	d_costs = 0;
	obj_val = 0;
	initial_assignment_count = 0;

	stepcounts = 0;
	steptimes = 0;

	d_vertices.row_assignments = new int[N];
	d_vertices.col_assignments = new int[N];
	d_vertices.row_covers = new int[N];
	d_vertices.col_covers = new int[N];

	d_row_duals = new double[N];
	d_col_duals = new double[N];
	d_col_slacks = new double[N];

	d_row_data.parents = new int[N];
	d_row_data.is_visited = new int[N];

	d_col_data.parents = new int[N];
	d_col_data.is_visited = new int[N];

	std::fill(d_vertices.row_assignments, d_vertices.row_assignments + N, -1);
	std::fill(d_vertices.col_assignments, d_vertices.col_assignments + N, -1);
	std::fill(d_vertices.row_covers, d_vertices.row_covers + N, 0);
	std::fill(d_vertices.col_covers, d_vertices.col_covers + N, 0);

	std::fill(d_row_data.parents, d_row_data.parents + N, -1);
	std::fill(d_row_data.is_visited, d_row_data.is_visited + N, 0);

	std::fill(d_col_data.parents, d_col_data.parents + N, -1);
	//	std::fill(d_col_data.is_visited, d_col_data.is_visited + N, 0);

	std::fill(d_row_duals, d_row_duals + N, 0);
	std::fill(d_col_duals, d_col_duals + N, 0);
	std::fill(d_col_slacks, d_col_slacks + N, INF);
}

SqLAP::~SqLAP()
{

	delete[] d_vertices.row_assignments;
	delete[] d_vertices.col_assignments;
	delete[] d_vertices.row_covers;
	delete[] d_vertices.col_covers;

	delete[] d_row_data.parents;
	delete[] d_row_data.is_visited;

	delete[] d_col_data.parents;
	delete[] d_col_data.is_visited;

	delete[] d_row_duals;
	delete[] d_col_duals;
	delete[] d_col_slacks;
}

// Executes Hungarian algorithm on the input cost matrix. Returns minimum cost.
int SqLAP::solve(double *_cost_matrix, double &_obj_val, int *_stepcounts, double *_steptimes, int &_initial_assignment_count)
{
	int step = 0;
	int total_count = 0;
	bool done = false;
	initial_assignment_count = 0;

	d_costs = _cost_matrix;
	stepcounts = _stepcounts;
	steptimes = _steptimes;

	while (!done)
	{
		total_count++;
		switch (step)
		{
		case 0:
			stepcounts[0]++;
			step = hungarianStep0(true);
			break;
		case 1:
			stepcounts[1]++;
			step = hungarianStep1(true);
			break;
		case 2:
			stepcounts[2]++;
			step = hungarianStep2(true);
			break;
		case 3:
			stepcounts[3]++;
			step = hungarianStep3(true);
			break;
		case 4:
			///counts[4]++;
			///step = hungarianStep4(true);
			break;
		case 5:
			stepcounts[5]++;
			step = hungarianStep5(true);
			break;
		case 6:
			stepcounts[6]++;
			step = hungarianStep6(true);
			break;
		case 100:
			done = true;
			break;
		}
	}

	_obj_val = obj_val;
	_initial_assignment_count = initial_assignment_count;

	return 0;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int SqLAP::hungarianStep0(bool count_time)
{
	double start = omp_get_wtime();

	rowReduction();	   // Kernel execution.
	columnReduction(); // Kernel execution.

	double end = omp_get_wtime();

	if (count_time)
		steptimes[0] += (end - start);

	return 1;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int SqLAP::hungarianStep1(bool count_time)
{
	double start = omp_get_wtime();

	computeInitialAssignments(); // Kernel execution.

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
		steptimes[1] += (end - start);

	return next;
}

// Function for checking optimality and constructing predicates and covers.
int SqLAP::hungarianStep2(bool count_time)
{
	int next = 3;

	double start = omp_get_wtime();

	std::fill(d_row_data.is_visited, d_row_data.is_visited + N, DORMANT);
	std::fill(d_col_data.is_visited, d_col_data.is_visited + N, DORMANT);
	std::fill(d_vertices.row_covers, d_vertices.row_covers + N, 0);
	std::fill(d_vertices.col_covers, d_vertices.col_covers + N, 0);

	std::fill(d_col_slacks, d_col_slacks + N, INF);

	int cover_count = executeCoverCount(); // Kernel execution.

	double end = omp_get_wtime();

	if (initial_assignment_count == 0)
		initial_assignment_count = cover_count;

	if (cover_count == N)
		next = 6;

	if (count_time)
		steptimes[2] += (end - start);

	return next;
}

// Function for finding minimum zero cover.
int SqLAP::hungarianStep3(bool count_time)
{
	int next;

	double start = omp_get_wtime();

	executeZeroCover(next); // execute zero cover algorithm.

	double end = omp_get_wtime();

	if (count_time)
		steptimes[3] += (end - start);

	return next;
}

// Function for updating the dual variables to increase the number of uncovered zeros.
int SqLAP::hungarianStep5(bool count_time)
{

	double start = omp_get_wtime();

	dualUpdate();

	double end = omp_get_wtime();

	if (count_time)
		steptimes[5] += (end - start);

	return 3;
}

// Function for calculating final objective function.
int SqLAP::hungarianStep6(bool count_time)
{

	double start = omp_get_wtime();

	finalCost();

	double end = omp_get_wtime();

	if (count_time)
		steptimes[6] += (end - start);

	return 100;
}

// Kernel for reducing the rows by subtracting row minimum from each row element.
void SqLAP::rowReduction(void)
{

	for (int rowid = 0; rowid < N; rowid++)
	{
		double min = INF;

		for (int colid = 0; colid < N; colid++)
		{
			double val = d_costs[rowid * N + colid];
			if (val < min)
			{
				min = val;
			}
		}

		d_row_duals[rowid] = min;
	}
}

// Kernel for reducing the column by subtracting column minimum from each column element.
void SqLAP::columnReduction(void)
{

	for (int colid = 0; colid < N; colid++)
	{
		double min = INF;

		for (int rowid = 0; rowid < N; rowid++)
		{
			double val = d_costs[rowid * N + colid];
			double dual = d_row_duals[rowid];
			if (val - dual < min)
			{
				min = val - dual;
			}
		}

		d_col_duals[colid] = min;
	}
}

// Kernel for calculating initial assignments.
void SqLAP::computeInitialAssignments(void)
{
	int *row_cover = new int[N];
	int *col_cover = new int[N];

	std::fill(row_cover, row_cover + N, 0);
	std::fill(col_cover, col_cover + N, 0);

	for (int colid = 0; colid < N; colid++)
	{
		if (col_cover[colid] == 0)
		{
			for (int rowid = 0; rowid < N; rowid++)
			{
				double cost = d_costs[rowid * N + colid];
				double rowdual = d_row_duals[rowid];
				double coldual = d_col_duals[colid];

				double slack = cost - rowdual - coldual;
				if ((slack > -EPSILON && slack < EPSILON) && row_cover[rowid] == 0)
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
int SqLAP::executeCoverCount(void)
{
	int cover_count = 0;

	for (int rowid = 0; rowid < N; rowid++)
	{
		d_row_data.is_visited[rowid] = ACTIVE;
		if (d_vertices.row_assignments[rowid] != -1)
		{
			d_vertices.row_covers[rowid] = 1;
			d_row_data.is_visited[rowid] = DORMANT;
			cover_count++;
		}
	}

	return cover_count;
}

// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
void SqLAP::executeZeroCover(int &next)
{
	next = 5;

	std::stack<int> q;

	for (int rowid = 0; rowid < N; rowid++)
	{
		if (d_row_data.is_visited[rowid] == ACTIVE)
		{
			q.push(rowid);
		}
	}

	while (!q.empty())
	{
		int rowid = q.top();
		q.pop();

		for (int colid = 0; colid < N; colid++)
		{
			double slack = d_costs[rowid * N + colid] - d_row_duals[rowid] - d_col_duals[colid];
			int nxt_rowid = d_vertices.col_assignments[colid];

			if (rowid != nxt_rowid && d_vertices.col_covers[colid] == 0)
			{
				if (slack < d_col_slacks[colid])
				{

					d_col_slacks[colid] = slack;
					d_col_data.parents[colid] = rowid;
				}

				if (d_col_slacks[colid] > -EPSILON && d_col_slacks[colid] < EPSILON)
				{

					if (nxt_rowid != -1)
					{
						d_row_data.parents[nxt_rowid] = colid; // update parent info

						d_vertices.row_covers[nxt_rowid] = 0;
						d_vertices.col_covers[colid] = 1;

						d_row_data.is_visited[nxt_rowid] = ACTIVE;
						q.push(nxt_rowid);
					}

					else
					{
						augment(colid);
						next = 2;
						return;
					}
				}
			}
		}

		d_row_data.is_visited[rowid] = VISITED;
	}
}

void SqLAP::augment(int colid)
{
	int cur_colid = colid;
	int cur_rowid = -1;

	while (cur_colid != -1)
	{
		cur_rowid = d_col_data.parents[cur_colid];

		d_vertices.row_assignments[cur_rowid] = cur_colid;
		d_vertices.col_assignments[cur_colid] = cur_rowid;

		cur_colid = d_row_data.parents[cur_rowid];
	}
}

// Kernel for updating the dual reduced costs in Step 5, without using atomic functions.
void SqLAP::dualUpdate(void)
{
	double delta = INF;
	int min_slack_colid = -1;

	for (int colid = 0; colid < N; colid++)
	{

		int cover = d_vertices.col_covers[colid];
		double slack = d_col_slacks[colid];

		if (cover == 0)
			if (slack < delta)
				delta = slack;
	}

	double theta = delta / 2;

	for (int rowid = 0; rowid < N; rowid++)
	{
		int cover = d_vertices.row_covers[rowid];

		if (cover == 0) // row in W
			d_row_duals[rowid] += theta;
		else
			// row in W'
			d_row_duals[rowid] -= theta;
	}

	for (int colid = 0; colid < N; colid++)
	{
		int cover = d_vertices.col_covers[colid];

		if (cover == 1) // col in W
			d_col_duals[colid] -= theta;
		else
		{
			// col in W'
			d_col_duals[colid] += theta;

			d_col_slacks[colid] -= delta;

			if (d_col_slacks[colid] > -EPSILON && d_col_slacks[colid] < EPSILON)
			{
				int act_rowid = d_col_data.parents[colid];
				d_row_data.is_visited[act_rowid] = ACTIVE;
			}
		}
	}
}

// Kernel for calculating the optimal assignment cost.
void SqLAP::finalCost(void)
{

	for (int rowid = 0; rowid < N; rowid++)
	{

		int colid = d_vertices.row_assignments[rowid];
		int id = rowid * N + colid;
		obj_val += d_costs[id];
	}

	double *rc = new double[N * N];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			rc[i * N + j] = d_costs[i * N + j] - d_row_duals[i] - d_col_duals[j];

	std::cout << "Obj: " << obj_val << std::endl;
	printHostMatrix(rc, N, N, "rc");

	delete[] rc;
}
