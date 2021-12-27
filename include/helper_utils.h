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

void printMemoryUsage(void);
void initialize(void);
void finalize(void);
void readFile(const char *filename);
void generateProblem(int problemsize, int costrange);
void printLog(int prno, int repetition, int costrange, int obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath);
void consoleOut(void);
void printDebugInfo(void);

void exclusiveSumScan(int *array, int size);

#endif