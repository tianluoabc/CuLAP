/*
 * Created by Ketan Date
 */

#ifndef _HUNGARIAN_ALGORITHM_H
#define _HUNGARIAN_ALGORITHM_H

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "variables.h"
#include "structures.h"
#include "helper_utils.h"
#include "exclusive_scan.h"
#include "reduction.h"
#include "functions_step_3.h"
#include "functions_step_4.h"


int solve (int* stepcounts, float *steptimes, int &init_assignments);
int hungarianStep0 (bool count_time);
int hungarianStep1 (bool count_time);
int hungarianStep2 (bool count_time);
int hungarianStep3 (bool count_time);
int hungarianStep4 (bool count_time);
int hungarianStep5 (bool count_time);
int hungarianStep6 (bool count_time);

__global__ void kernel_rowReduction (int *d_costs, int N);
__global__ void kernel_columnReduction (int *d_costs, int N);
__global__ void kernel_computeInitialAssignments (char *d_masks, int *d_costs, int *d_row_lock, int *d_col_lock, int N); 
__global__ void kernel_populateAssignments ( short *d_row_assignments, short *d_col_assignments, int *d_row_covers, char *d_masks, int *d_costs, int N);
__global__ void kernel_dualUpdate_1_nonAtomic (int *d_min_val, int *d_costs, int *d_row_cover, int *d_col_cover, int N);
__global__ void kernel_dualUpdate_1 (int *d_min_val, int *d_costs, int *d_row_cover, int *d_col_cover, int N);
__global__ void kernel_dualUpdate_2 (int d_min_val, char *d_masks, int *d_costs, int *d_row_cover, int *d_col_cover, short *d_row_visited, int N);
__global__ void kernel_finalCost (int *d_obj_val, int *d_costs, char *d_masks, int N);

#endif