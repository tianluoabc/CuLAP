all: exclusive_scan.o sfmt.o functions_step_3.o functions_step_4.o helper_utils.o hungarian_algorithm.o main.o reduction.o
	g++ o/exclusive_scan.o o/sfmt.o o/functions_step_4.o o/helper_utils.o o/hungarian_algorithm.o o/main.o o/reduction.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o cuLAP

exclusive_scan.o: exclusive_scan.cu
	nvcc -arch=sm_75 -c exclusive_scan.cu -o o/exclusive_scan.o

functions_step_3.o: functions_step_3.cu
	nvcc -arch=sm_75 -c functions_step_3.cu -o o/functions_step_3.o

functions_step_4.o: functions_step_4.cu
	nvcc -arch=sm_75 -c functions_step_4.cu -o o/functions_step_4.o

helper_utils.o: helper_utils.cu
	nvcc -arch=sm_75 -c helper_utils.cu -o o/helper_utils.o

hungarian_algorithm.o: hungarian_algorithm.cu
	nvcc -lineinfo -O3 -arch=sm_75 -c hungarian_algorithm.cu -o o/hungarian_algorithm.o

main.o: main.cu
	nvcc -arch=sm_75 -c main.cu -o o/main.o

reduction.o: reduction.cu
	nvcc -arch=sm_75 -c reduction.cu -o o/reduction.o

sfmt.o: sfmt.cpp
	g++ -I/usr/local/cuda/include -c sfmt.cpp -o o/sfmt.o

clean:
	rm -rf o/*.o