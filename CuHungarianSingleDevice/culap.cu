/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*	   
*/

#include "culap.h"

CuLAP::CuLAP(int _size, int _batchsize, int _devid) {

	N = _size;
	N2 = N * N;

	SP = _batchsize;
	devid = _devid;

	M = 0;

	prevstep = 0;

	flag = false;

	initial_assignment_count = 0;

	stepcounts = new int[7];
	steptimes = new float[9];

	h_obj_val_primal = new float[SP];
	h_obj_val_dual = new float[SP];

	h_row_duals = new float[SP * N];
	h_col_duals = new float[SP * N];

	h_row_assignments = new int[SP * N];

	d_obj_val_primal = 0;

	d_obj_val_dual = 0;

	dual_update_itns = 0;

}

CuLAP::~CuLAP() {

	delete[] stepcounts;
	delete[] steptimes;
	delete[] h_obj_val_primal;
	delete[] h_obj_val_dual;
	delete[] h_row_assignments;
	delete[] h_row_duals;
	delete[] h_col_duals;
}

// Helper function for initializing global variables and arrays on a single host.
void CuLAP::initializeDevice(float *h_cost_matrix) {

	cudaSetDevice(devid);

	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.row_assignments), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_row_assignment");
	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.col_assignments), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_col_assignment");
	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.row_covers), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_row_covers");
	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.col_covers), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_col_covers");

	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.row_duals), SP * N * sizeof(float)), "error in cudaMalloc CuLAP::initializeDevice::d_row_duals");
	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.col_duals), SP * N * sizeof(float)), "error in cudaMalloc CuLAP::initializeDevice::d_col_duals");

	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.col_slacks), SP * N * sizeof(float)), "error in cudaMalloc CuLAP::initializeDevice::d_col_slacks");

	cudaSafeCall(cudaMalloc((void**) (&d_row_data_dev.is_visited), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_row_data.is_visited");
	cudaSafeCall(cudaMalloc((void**) (&d_col_data_dev.is_visited), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_col_data.is_visited");

	cudaSafeCall(cudaMalloc((void**) (&d_row_data_dev.parents), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_row_data.parents");
	cudaSafeCall(cudaMalloc((void**) (&d_row_data_dev.children), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_row_data.children");
	cudaSafeCall(cudaMalloc((void**) (&d_col_data_dev.parents), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_col_data.parents");
	cudaSafeCall(cudaMalloc((void**) (&d_col_data_dev.children), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_col_data.children");

	cudaSafeCall(cudaMalloc((void**)(&d_costs_dev.elements), SP * N * N * sizeof(float)), "error in cudaMalloc CuLAP::initializeDevice::d_costs");

	cudaSafeCall(cudaMalloc((void**)(&d_obj_val_primal), SP * sizeof(float)), "error in cudaMalloc CuLAP::initializeDevice::d_obj_val_primal");
	cudaSafeCall(cudaMalloc((void**)(&d_obj_val_dual), SP * sizeof(float)), "error in cudaMalloc CuLAP::initializeDevice::d_obj_val_dual");

	cudaSafeCall(cudaMemset(d_vertices_dev.row_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset CuLAP::initializeDevice::d_col_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev.row_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset CuLAP::initializeDevice::d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset CuLAP::initializeDevice::d_col_covers");

	cudaSafeCall(cudaMemset(d_vertices_dev.row_duals, 0, SP * N * sizeof(float)), "Error in cudaMemset CuLAP::initializeDevice::d_row_duals");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_duals, 0, SP * N * sizeof(float)), "Error in cudaMemset CuLAP::initializeDevice::d_col_duals");

	cudaSafeCall(cudaMemcpy(d_costs_dev.elements, h_cost_matrix, SP * N * N * sizeof(float), cudaMemcpyHostToDevice), "error in cudaMemcpy CuLAP::initializeDevice::d_costs");

}

// Helper function for finalizing global variables and arrays on a single host.
void CuLAP::finalizeDevice(void) {

	cudaSetDevice(devid);

	cudaSafeCall(cudaMemcpy(h_row_assignments, d_vertices_dev.row_assignments, SP * N * sizeof(int), cudaMemcpyDeviceToHost), "error in cudaMemcpy CuLAP::finalizeDevice::d_row_assignments");
	cudaSafeCall(cudaMemcpy(h_obj_val_primal, d_obj_val_primal, SP * sizeof(float), cudaMemcpyDeviceToHost), "error in cudaMemcpy CuLAP::finalizeDevice::d_obj_val_primal");
	cudaSafeCall(cudaMemcpy(h_obj_val_dual, d_obj_val_dual, SP * sizeof(float), cudaMemcpyDeviceToHost), "error in cudaMemcpy CuLAP::finalizeDevice::d_obj_val_dual");
	cudaSafeCall(cudaMemcpy(h_row_duals, d_vertices_dev.row_duals, SP * N * sizeof(float), cudaMemcpyDeviceToHost), "error in cudaMemcpy CuLAP::finalizeDevice::d_row_duals");
	cudaSafeCall(cudaMemcpy(h_col_duals, d_vertices_dev.col_duals, SP * N * sizeof(float), cudaMemcpyDeviceToHost), "error in cudaMemcpy CuLAP::finalizeDevice::d_col_duals");

	cudaSafeCall(cudaFree(d_vertices_dev.row_assignments), "Error in cudaFree CuLAP::finalizeDevice::d_row_assignment");
	cudaSafeCall(cudaFree(d_vertices_dev.col_assignments), "Error in cudaFree CuLAP::finalizeDevice::d_col_assignment");
	cudaSafeCall(cudaFree(d_vertices_dev.row_covers), "Error in cudaFree CuLAP::finalizeDevice::d_row_covers");
	cudaSafeCall(cudaFree(d_vertices_dev.col_covers), "Error in cudaFree CuLAP::finalizeDevice::d_col_covers");

	cudaSafeCall(cudaFree(d_vertices_dev.row_duals), "Error in cudaFree CuLAP::finalizeDevice::d_row_duals");
	cudaSafeCall(cudaFree(d_vertices_dev.col_duals), "Error in cudaFree CuLAP::finalizeDevice::d_col_duals");

	cudaSafeCall(cudaFree(d_vertices_dev.col_slacks), "Error in cudaFree CuLAP::finalizeDevice::d_col_slacks");

	cudaSafeCall(cudaFree(d_row_data_dev.is_visited), "Error in cudaFree CuLAP::finalizeDevice::d_row_data.is_visited");
	cudaSafeCall(cudaFree(d_col_data_dev.is_visited), "Error in cudaFree CuLAP::finalizeDevice::d_col_data.is_visited");

	cudaSafeCall(cudaFree(d_row_data_dev.parents), "Error in cudaFree CuLAP::finalizeDevice::d_row_data.parents");
	cudaSafeCall(cudaFree(d_row_data_dev.children), "Error in cudaFree CuLAP::finalizeDevice::d_row_data.children");
	cudaSafeCall(cudaFree(d_col_data_dev.parents), "Error in cudaFree CuLAP::finalizeDevice::d_col_data.parents");
	cudaSafeCall(cudaFree(d_col_data_dev.children), "Error in cudaFree CuLAP::finalizeDevice::d_col_data.children");

	cudaSafeCall(cudaFree(d_costs_dev.elements), "error in cudaFree CuLAP::finalizeDevice::d_costs");

	cudaSafeCall(cudaFree(d_obj_val_primal), "error in cudaFree CuLAP::finalizeDevice::d_obj_val_primal");
	cudaSafeCall(cudaFree(d_obj_val_dual), "error in cudaFree CuLAP::finalizeDevice::d_obj_val_dual");

	cudaDeviceReset();
}

// Executes Hungarian algorithm on the input cost matrix. Returns minimum cost.
int CuLAP::solve(float *h_cost_matrix) {

	initializeDevice(h_cost_matrix);

	int step = 0;
	int total_count = 0;
	bool done = false;
	prevstep = -1;

	std::fill(stepcounts, stepcounts + 7, 0);
	std::fill(steptimes, steptimes + 9, 0);

	while (!done) {
		total_count++;
		switch (step) {
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
			stepcounts[4]++;
			step = hungarianStep4(true);
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

	finalizeDevice();

	return 0;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int CuLAP::hungarianStep0(bool count_time) {

	float start = omp_get_wtime();

	initialReduction(d_costs_dev, d_vertices_dev, SP, N, devid);

	float end = omp_get_wtime();

	if (count_time)
		steptimes[0] += (end - start);

	prevstep = 0;

	return 1;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int CuLAP::hungarianStep1(bool count_time) {

	float start = omp_get_wtime();

	computeInitialAssignments(d_costs_dev, d_vertices_dev, SP, N, devid);

	float mid = omp_get_wtime();

	int next = 2;

	while (true) {

		initial_assignment_count = 0;

		if ((next = hungarianStep2(false)) == 6)
			break;

		if ((next = hungarianStep3(false)) == 5)
			break;

		hungarianStep4(false);
	}

	float end = omp_get_wtime();
	if (count_time) {
		steptimes[1] += (mid - start);
		steptimes[2] += (end - mid);
	}
	prevstep = 1;

	return next;
}

// Function for checking optimality and constructing predicates and covers.
int CuLAP::hungarianStep2(bool count_time) {

	float start = omp_get_wtime();

	int cover_count = computeRowCovers(d_vertices_dev, d_row_data_dev, d_col_data_dev, SP, N, devid);

	if (initial_assignment_count == 0)
		initial_assignment_count = cover_count;

	int next = (cover_count == SP * N) ? 6 : 3;

	float end = omp_get_wtime();

	if (count_time)
		steptimes[3] += (end - start);

	prevstep = 2;

	return next;
}

// Function for building alternating tree rooted at unassigned rows.
int CuLAP::hungarianStep3(bool count_time) {

	float start = omp_get_wtime();


	///////////////////////////////////////////////////////////////

	float mid = omp_get_wtime();

	///////////////////////////////////////////////////////////////

	int next;

	bool h_flag = false;

	executeZeroCover(d_costs_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, &h_flag, SP, N, devid); // execute zero cover algorithm.

	next = h_flag ? 4 : 5;

	///////////////////////////////////////////////////////////////

	float end = omp_get_wtime();

	if (count_time) {
		steptimes[4] += (mid - start);
		steptimes[5] += (end - mid);
	}

	prevstep = 3;

	return next;
}

// Function for augmenting the solution along multiple node-disjoint alternating trees.
int CuLAP::hungarianStep4(bool count_time) {

	float start = omp_get_wtime();

///////////////////////////////////////////////////////////////

	reversePass(d_row_data_dev, d_col_data_dev, SP, N, devid); // execute reverse pass of the maximum matching algorithm.

	augmentationPass(d_vertices_dev, d_row_data_dev, d_col_data_dev, SP, N, devid); // execute augmentation pass of the maximum matching algorithm.

///////////////////////////////////////////////////////////////

	float end = omp_get_wtime();

	if (count_time)
		steptimes[6] += (end - start);

	prevstep = 4;

	return 2;
}

// Function for updating dual solution to introduce new zero-cost arcs.
int CuLAP::hungarianStep5(bool count_time) {

	float start = omp_get_wtime();

	dualUpdate(d_vertices_dev, d_row_data_dev, d_col_data_dev, SP, N, devid);

	dual_update_itns++;

	float end = omp_get_wtime();

	if (count_time)
		steptimes[7] += (end - start);

	prevstep = 5;

	return 3;
}

// Function for calculating primal and dual objective values at optimality.
int CuLAP::hungarianStep6(bool count_time) {

	float start = omp_get_wtime();

	calcObjValPrimal(d_obj_val_primal, d_costs_dev.elements, d_vertices_dev, SP, N, devid);

	calcObjValDual(d_obj_val_dual, d_vertices_dev, SP, N, devid);

	float end = omp_get_wtime();

	if (count_time)
		steptimes[8] += (end - start);

	prevstep = 6;

	return 100;
}

// Function for getting optimal assignment vector for subproblem spId.
void CuLAP::getAssignmentVector(int *out, int spId) {

	try {
		memcpy(out, &h_row_assignments[spId * N], N * sizeof(int));
	}

	catch (...) {
		std::cerr << "Cannot access assignment array index " << spId << std::endl;

		throw;
	}
}

// Function for getting optimal row dual vector for subproblem spId.
void CuLAP::getRowDualVector(float *out, int spId) {

	try {
		memcpy(out, &h_row_duals[spId * N], N * sizeof(float));
	}

	catch (...) {
		std::cerr << "Cannot access row dual array index " << spId << std::endl;

		throw;
	}
}

// Function for getting optimal col dual vector for subproblem spId.
void CuLAP::getColDualVector(float *out, int spId) {

	try {
		memcpy(out, &h_col_duals[spId * N], N * sizeof(float));
	}

	catch (...) {
		std::cerr << "Cannot access col dual array index " << spId << std::endl;

		throw;
	}
}

// Function for getting optimal primal objective value for subproblem spId.
float CuLAP::getPrimalObjectiveValue(int spId) {

	try {
		return h_obj_val_primal[spId];
	}

	catch (...) {
		std::cerr << "Cannot access primal objective array index " << spId << std::endl;

		throw;
	}
}

// Function for getting optimal dual objective value for subproblem spId.
float CuLAP::getDualObjectiveValue(int spId) {

	try{
		return h_obj_val_dual[spId];
	}
	
	catch (...) {
		std::cerr << "Cannot access dual objective array index " << spId << std::endl;

		throw;
	}
	
}

