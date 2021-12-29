/*
 * LinearAssignmentProblem.h
 *
 *  Created on: Oct 30, 2014
 *      Author: ketandat
 */



#ifndef LINEARASSIGNMENTPROBLEM_H_
#define LINEARASSIGNMENTPROBLEM_H_

#include <omp.h>
#include "structures.h"
#include "variables.h"
#include "functions_step_0.h"
#include "functions_step_1.h"
#include "functions_step_2.h"
#include "functions_step_3.h"
#include "functions_step_4.h"
#include "functions_step_5.h"

class LinearAssignmentProblem {

	int N;
	long N2;
	long M; // total number of zero cost edges on a single host.

	int initial_assignment_count;
	int *stepcounts;
	double *steptimes;

	int numdev;
	int prevstep;

	bool flag;

	double obj_val;

	Matrix h_costs, *d_costs_dev;
	Vertices h_vertices, *d_vertices_dev;
	CompactEdges *d_edges_csr_dev;
	VertexData *d_row_data_dev, *d_col_data_dev;

public:
	LinearAssignmentProblem(int _size, int _numdev);
	virtual ~LinearAssignmentProblem();

	int solve(double *cost_matrix, double &obj_val);

	void getAssignments(int *_row_assignments);
	void getStepTimes(double *_steptimes);
	void getStepCounts(int *_stepcounts);

private:

	void initializeDevice(unsigned int devid);
	void finalizeDev(unsigned int devid);

	int hungarianStep0(bool count_time);
	int hungarianStep1(bool count_time);
	int hungarianStep2(bool count_time);
	int hungarianStep3(bool count_time);
	int hungarianStep4(bool count_time);
	int hungarianStep5(bool count_time);
	int hungarianStep6(bool count_time);


};

#endif /* LINEARASSIGNMENTPROBLEM_H_ */
