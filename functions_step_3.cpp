/*
 * Created by Ketan Date
 */

#include "include/functions_step_3.h"

void printVertexData(void);

// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
void executeZeroCover(int &next)
{
	next = 5;
	int total_blocks = 0;

	if (d_edges_csr.nbrs == NULL)
		compactEdges();

	std::stack<int> q;

	for (int i = 0; i < N; i++)
		if (d_row_data.is_visited[i] == ACTIVE)
			q.push(i);

	while (!q.empty())
	{
		int src = q.top();
		q.pop();

		int start = d_edges_csr.ptrs[src];
		int end = d_edges_csr.ptrs[src + 1];
		int size = end - start;

		int *alloc_start = &d_edges_csr.nbrs[start];
		int *alloc_end = alloc_start + size;

		if (d_row_data.is_visited[src] == ACTIVE && size > 0)
		{
			int *ptr = alloc_start;

			while (ptr != alloc_end)
			{
				int colid = *ptr;
				int newrow = d_vertices.col_assignments[colid];

				if (src != newrow && d_vertices.col_covers[colid] == 0)
				{
					d_col_data.parents[colid] = src;

					if (newrow != -1)
					{
						d_row_data.parents[newrow] = colid; // update parent info

						d_vertices.row_covers[newrow] = 0;
						d_vertices.col_covers[colid] = 1;

						if (d_row_data.is_visited[newrow] == DORMANT)
						{
							d_row_data.is_visited[newrow] = ACTIVE;
							q.push(newrow);
						}
					}

					else
					{
						augment(colid);
						next = 2;
						return;
					}
				}

				ptr++;
			}
		}

		d_row_data.is_visited[src] = VISITED;
	}

	if (next == 5)
	{
		delete[] d_edges_csr.nbrs;
		delete[] d_edges_csr.ptrs;
		d_edges_csr.nbrs = NULL;
		d_edges_csr.ptrs = NULL;
	}
}

void augment(int colid)
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

void compactEdges(void)
{

	M = 0;

	for (int i = 0; i < N2; i++)
	{
		if (d_edges.costs[i] == 0)
			M++;
	}

	d_edges_csr.nbrs = new int[M];
	d_edges_csr.ptrs = new int[N + 1];

	M = 0;
	for (int rowid = 0; rowid < N; rowid++)
	{
		d_edges_csr.ptrs[rowid] = M;
		for (int colid = 0; colid < N; colid++)
		{
			int id = rowid * N + colid;
			if (d_edges.costs[id] == 0)
			{
				d_edges_csr.nbrs[M] = colid;

				M++;
			}
		}
	}

	d_edges_csr.ptrs[N] = M;
}