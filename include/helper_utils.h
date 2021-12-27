/*
 * Developed by Ketan Date
 */

#ifndef _HELPER_UTILS_H
#define _HELPER_UTILS_H

#include <iostream>
#include <fstream>
#include "structures.h"
#include "variables.h"
#include "sfmt.h"

void printMemoryUsage (void);
void initialize (void);
void finalize(void);
void readFile(double *cost_matrix, const char *filename);

void generateProblem(int *cost_matrix, int N, int costrange);
void printLog(int prno, int repetition, int N, int costrange, double obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath);
void consoleOut(void);
void printDebugInfo(void);

void printHostArray(long *h_array, int size, const char *name);
void printHostMatrix(double *h_matrix, int rowsize, int colsize, const char *name);
void printHostArray(int *h_array, int size, const char *name);
void printHostArray(double *h_array, int size, const char *name);

#endif
