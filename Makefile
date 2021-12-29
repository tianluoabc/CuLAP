CUDAFLAGS = -arch=sm_75 -w -std=c++14 -O3 

all: sfmt.o f_culap.o culap.o helper_utils.o main.o
	g++ o/sfmt.o o/f_culap.o o/culap.o o/helper_utils.o o/main.o -L/usr/local/cuda/lib64 -lcudart -lgomp -o cuLAP

functions_step_5.o: functions_step_5.cu
	nvcc $(CUDAFLAGS) functions_step_5.cu -o o/functions_step_5.o

helper_utils.o: helper_utils.cpp
	g++ -I/usr/local/cuda/include -c helper_utils.cpp -o o/helper_utils.o

culap.o: culap.cu
	nvcc $(CUDAFLAGS) -c culap.cu -o o/culap.o

f_culap.o: f_culap.cu
	nvcc $(CUDAFLAGS) -c f_culap.cu -o o/f_culap.o

main.o: src/main.cu
	nvcc $(CUDAFLAGS) -c src/main.cu -o o/main.o

sfmt.o: sfmt.cpp
	g++ -I/usr/local/cuda/include -c sfmt.cpp -o o/sfmt.o

clean:
	rm -rf o/*.o cuLAP