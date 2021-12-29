all: sfmt.o functions_step_0.o functions_step_1.o functions_step_2.o functions_step_3.o functions_step_4.o functions_step_5.o helper_utils.o LinearAssignmentProblem.o main.o
	g++ o/sfmt.o o/functions_step_0.o o/functions_step_1.o o/functions_step_2.o o/functions_step_3.o o/functions_step_4.o o/functions_step_5.o o/helper_utils.o o/LinearAssignmentProblem.o o/main.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o cuLAP

functions_step_0.o: functions_step_0.cu
	nvcc -arch=sm_75 -c functions_step_0.cu -o o/functions_step_0.o

functions_step_1.o: functions_step_1.cu
	nvcc -arch=sm_75 -c functions_step_1.cu -o o/functions_step_1.o

functions_step_2.o: functions_step_2.cu
	nvcc -arch=sm_75 -c functions_step_2.cu -o o/functions_step_2.o

functions_step_3.o: functions_step_3.cu
	nvcc -arch=sm_75 -c functions_step_3.cu -o o/functions_step_3.o

functions_step_4.o: functions_step_4.cu
	nvcc -arch=sm_75 -c functions_step_4.cu -o o/functions_step_4.o

functions_step_5.o: functions_step_5.cu
	nvcc -arch=sm_75 -c functions_step_5.cu -o o/functions_step_5.o

helper_utils.o: helper_utils.cpp
	g++ -I/usr/local/cuda/include -c helper_utils.cpp -o o/helper_utils.o

LinearAssignmentProblem.o: LinearAssignmentProblem.cpp
	g++ -I/usr/local/cuda/include -c LinearAssignmentProblem.cpp -o o/LinearAssignmentProblem.o

main.o: src/main.cpp
	g++ -I/usr/local/cuda/include -c src/main.cpp -o o/main.o

sfmt.o: sfmt.cpp
	g++ -I/usr/local/cuda/include -c sfmt.cpp -o o/sfmt.o

clean:
	rm -rf o/*.o