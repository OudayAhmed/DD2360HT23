#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType* in1, DataType* in2, DataType* out, int len) {
	//@@ Insert code to implement vector addition here
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
		out[i] = in1[i] + in2[i];
}

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char** argv) {

	int inputLength;
	DataType* hostInput1;
	DataType* hostInput2;
	DataType* hostOutput;
	DataType* resultRef;
	DataType* deviceInput1;
	DataType* deviceInput2;
	DataType* deviceOutput;

	//@@ Insert code below to read in inputLength from args
	if (argc == 2) {
		inputLength = std::atoi(argv[1]);
		printf("The input length is %d\n", inputLength);
	}
	else {
		printf("The input length is missing.\n");
	}

	//@@ Insert code below to allocate Host memory for input and output
	hostInput1 = new double[inputLength];
	hostInput2 = new double[inputLength];
	hostOutput = new double[inputLength];

	//@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distribution(0.0, 100.0);
	for (int i = 0; i < inputLength; i++) {
		hostInput1[i] = distribution(gen);
		hostInput2[i] = distribution(gen);
	}

	for (int i = 0; i < inputLength; i++)
		hostOutput[i] = hostInput1[i] + hostInput2[i];

	resultRef = new DataType[inputLength];

	//@@ Insert code below to allocate GPU memory here
	cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
	cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
	cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));

	//@@ Insert code to below to Copy memory to the GPU here

	double iStart01 = cpuSecond();
	cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
	double iElaps01 = cpuSecond() - iStart01;

	//@@ Initialize the 1D grid and block dimensions here
	int Db = 512;
	int Dg = (inputLength + Db - 1) / Db;

	//@@ Launch the GPU Kernel here
	double iStart02 = cpuSecond();
	vecAdd << <Dg, Db >> > (deviceInput1, deviceInput2, deviceOutput, inputLength);
	cudaDeviceSynchronize();
	double iElaps02 = cpuSecond() - iStart02;


	//@@ Copy the GPU memory back to the CPU here
	double iStart03 = cpuSecond();
	cudaMemcpy(resultRef, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
	double iElaps03 = cpuSecond() - iStart03;

	//@@ Insert code below to compare the output with the reference
	for (int i = 0; i < inputLength; i++)
		if (abs(hostOutput[i] - resultRef[i]) > 1e-10)
			printf("Values at index %d are not equal within 1e-10.", i);

	std::cout << "The data copy time from host to device is" << iElaps01 << std::endl;
	std::cout << "The CUDA kernel execution time is " << iElaps02 << std::endl;
	std::cout << "The data copy time from device to host is" << iElaps03 << std::endl;

	//@@ Free the GPU memory here
	cudaFree(deviceOutput);
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);

	//@@ Free the CPU memory here
	delete[] hostInput1;
	delete[] hostInput2;
	delete[] hostOutput;
	delete[] resultRef;

	return 0;
}
