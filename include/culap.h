/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*
*/

#include <omp.h>
#include "f_culap.h"
#include "d_structs.h"
#include "d_vars.h"

#ifndef CUALP_H_
#define CUALP_H_

class CuLAP {

	int N;
	long N2;
	long M; // total number of zero cost edges on a single host.
	int SP;
	int devid;

	int initial_assignment_count;
	int *stepcounts;
	float *steptimes;

	int prevstep;

	bool flag;

	float *d_obj_val_primal, *d_obj_val_dual;

	float *h_obj_val_primal, *h_obj_val_dual;

	int *h_row_assignments;

	float *h_row_duals, *h_col_duals;

	Matrix d_costs_dev;
	Vertices d_vertices_dev;
	CompactEdges d_edges_csr_dev;
	VertexData d_row_data_dev, d_col_data_dev;

	int dual_update_itns;

public:
	CuLAP(int _size, int _batchsize, int _devid);
	virtual ~CuLAP();

	int solve(float *h_cost_matrix);

	void getAssignmentVector(int *out, int spId);
	void getStepTimes(float *_steptimes);
	void getStepCounts(int *_stepcounts);

	float getPrimalObjectiveValue(int spId);
	float getDualObjectiveValue(int spId);

	void getRowDualVector(float *out, int spId);
	void getColDualVector(float *out, int spId);

private:

	void initializeDevice(float *h_cost_matrix);
	void finalizeDevice(void);

	int hungarianStep0(bool count_time);
	int hungarianStep1(bool count_time);
	int hungarianStep2(bool count_time);
	int hungarianStep3(bool count_time);
	int hungarianStep4(bool count_time);
	int hungarianStep5(bool count_time);
	int hungarianStep6(bool count_time);


};

#endif /* CUALP_H_ */
