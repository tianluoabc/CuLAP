/*
 * Created by Ketan Date
 */

#ifndef _HUNGARIAN_ALGORITHM_H
#define _HUNGARIAN_ALGORITHM_H

#include <ctime>
#include <iostream>
#include <fstream>
#include "variables.h"
#include "structures.h"
#include "helper_utils.h"
#include "functions_step_3.h"
#include <omp.h>


int solve (int* stepcounts, double *steptimes, int &init_assignments);
int hungarianStep0 (bool count_time);
int hungarianStep1 (bool count_time);
int hungarianStep2 (bool count_time);
int hungarianStep3 (bool count_time);
//int hungarianStep4 (bool count_time);
int hungarianStep5 (bool count_time);
int hungarianStep6 (bool count_time);

void rowReduction (int *d_costs, int N);
void columnReduction (int *d_costs, int N);
void computeInitialAssignments (int *d_masks, int *d_costs, int N);
int populateAssignments ( int *d_row_assignments, int *d_col_assignments, int *d_row_covers, int *d_masks, int *d_costs, int N);
void initialzeVertices(void);
void dualUpdate_1 (int& d_min_val, int *d_costs, int *d_row_cover, int *d_col_cover, int N);

void dualUpdate_2 (int d_min_val, int *d_masks, int *d_costs, int *d_row_cover, int *d_col_cover, int N);
void finalCost (int& d_obj_val, int *d_costs, int *d_row_assignments, int N);

#endif