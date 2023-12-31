#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <sys/time.h>

#define DataType double


// Compute C = A * B
__global__ void gemm(DataType* A, DataType* B, DataType* C, int numARows,
    int numAColumns, int numBRows, int numBColumns) {
    //@@ Insert code to implement matrix multiplication here
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numBColumns) {
        DataType sum = 0.0;
        for (int i = 0; i < numAColumns; i++)
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        C[row * numBColumns + col] = sum;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char** argv) {

    DataType* hostA; // The A matrix
    DataType* hostB; // The B matrix
    DataType* hostC; // The output C matrix
    DataType* resultRef; // The reference result
    DataType* deviceA;
    DataType* deviceB;
    DataType* deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
    if (argc == 4) {
        numARows = std::atoi(argv[1]);
        numAColumns = std::atoi(argv[2]);
        numBColumns = std::atoi(argv[3]);
        numBRows = numAColumns;
        numCRows = numARows;
        numCColumns = numBColumns;
    }
    else {
        printf("The number of arguments must be four.\n");
    }

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    //@@ Insert code below to allocate Host memory for input and output
    hostA = new DataType[numARows * numAColumns];
    hostB = new DataType[numBRows * numBColumns];
    hostC = new DataType[numCRows * numCColumns];

    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 100.0);
    for (int i = 0; i < numARows * numAColumns; i++)
        hostA[i] = distribution(gen);

    for (int i = 0; i < numBRows * numBColumns; i++)
        hostB[i] = distribution(gen);

    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            DataType sum = 0.0;
            for (int k = 0; k < numAColumns; k++)
                sum += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            hostC[i * numCColumns + j] = sum;
        }
    }
    resultRef = new DataType[numCRows * numCColumns];

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
    cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
    cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here
    double iStart01 = cpuSecond();
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    double iElaps01 = cpuSecond() - iStart01;

    //@@ Initialize the grid and block dimensions here
    dim3 block(32, 32);
    dim3 grid((numBColumns + block.x - 1) / block.x, (numARows + block.y - 1) / block.y);

    //@@ Launch the GPU Kernel here
    double iStart02 = cpuSecond();
    gemm << <grid, block >> > (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    double iElaps02 = cpuSecond() - iStart02;

    //@@ Copy the GPU memory back to the CPU here
    double iStart03 = cpuSecond();
    cudaMemcpy(resultRef, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    double iElaps03 = cpuSecond() - iStart03;

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < numCRows * numCColumns; i++)
        if (abs(resultRef[i] - hostC[i]) > 1e-10)
            printf("Values at index %d are not equal within 1e-10.", i);

    std::cout << "The data copy time from host to device is" << iElaps01 << std::endl;
    std::cout << "The CUDA kernel execution time is " << iElaps02 << std::endl;
    std::cout << "The data copy time from device to host is" << iElaps03 << std::endl;

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //@@ Free the CPU memory here
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    delete[] resultRef;

    return 0;
}
