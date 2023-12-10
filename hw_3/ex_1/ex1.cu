#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int* input, unsigned int* bins,
    unsigned int num_elements,
    unsigned int num_bins) {

    //@@ Insert code below to compute histogram of input using shared memory and atomics
    __shared__ unsigned int shared_bins[NUM_BINS];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x < num_bins)
        shared_bins[threadIdx.x] = 0;
    __syncthreads();
    if (tid < num_elements)
        atomicAdd(&shared_bins[input[tid]], 1);
    __syncthreads();
    if (threadIdx.x == 0)
      for (int i = 0; i < NUM_BINS; i++)
        atomicAdd(&bins[i], shared_bins[i]);
}

__global__ void convert_kernel(unsigned int* bins, unsigned int num_bins) {

    //@@ Insert code below to clean up bins that saturate at 127
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_bins)
        if (bins[tid] > 127)
            bins[tid] = 127;
}


int main(int argc, char** argv) {

    int inputLength;
    unsigned int* hostInput;
    unsigned int* hostBins;
    unsigned int* resultRef;
    unsigned int* deviceInput;
    unsigned int* deviceBins;

    //@@ Insert code below to read in inputLength from args
    if (argc == 2) {
        inputLength = std::atoi(argv[1]);
        printf("The input length is %d\n", inputLength);
    }
    else {
        printf("The input length is missing.\n");
        return -1;
    }

    //@@ Insert code below to allocate Host memory for input and output
    hostInput = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    for (int i = 0; i < inputLength; i++)
        hostInput[i] = std::rand() % NUM_BINS;

    //@@ Insert code below to create reference result in CPU
    resultRef = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < NUM_BINS; i++) {
        resultRef[i] = 0;
        hostBins[i] = 0;
    }

    for (int i = 0; i < inputLength; i++) {
        if (hostBins[hostInput[i]] == 127)
            continue;
        hostBins[hostInput[i]]++;
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    int Db = 1024;
    int Dg = (inputLength + Db - 1) / Db;

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<Dg, Db>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    cudaDeviceSynchronize();

    //@@ Initialize the second grid and block dimensions here
    Db = 1024;
    Dg = (NUM_BINS + Db - 1) / Db;

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<Dg, Db >>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(resultRef, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < NUM_BINS; i++)
        if (resultRef[i] != hostBins[i])
            printf("Values at index %d are not equal.", i);

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}