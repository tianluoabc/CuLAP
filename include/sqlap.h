/*
 * SqLAP.h
 *
 *  Created on: Jun 15, 2015
 *      Author: date2
 */

#ifndef SQLAP_H_
#define SQLAP_H_

#include "variables.h"
#include "structures.h"
#include "helper_utils.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <queue>
#include <stack>

class SqLAP {

	double obj_val;
	int N;
	int N2;
	int M;

	int initial_assignment_count;
	int *stepcounts;
	double *steptimes;

	double *d_costs;

	double *d_col_slacks;
	double *d_row_duals;
	double *d_col_duals;

	Vertices d_vertices;
	VertexData d_row_data, d_col_data;

public:
	SqLAP(int _size);
	virtual ~SqLAP();

	int solve (double *_cost_matrix, double &_obj_val, int *_stepcounts, double *_steptimes, int &_initial_assignment_count);

private:

	int hungarianStep0 (bool count_time);
	int hungarianStep1 (bool count_time);
	int hungarianStep2 (bool count_time);
	int hungarianStep3 (bool count_time);
	//int hungarianStep4 (bool count_time);
	int hungarianStep5 (bool count_time);
	int hungarianStep6 (bool count_time);

	void rowReduction (void);
	void columnReduction (void);
	void computeInitialAssignments (void);
	int executeCoverCount (void);

	void executeZeroCover(int &next);
	void augment (int colid);

	void dualUpdate (void);
	void finalCost (void);
};


#endif /* SQLAP_H_ */
