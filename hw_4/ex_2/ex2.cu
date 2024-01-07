#include <iostream>
#include <stdio.h>
#include <sys/time.h> 
#include <chrono>
#include <cstdlib>

#define DataType float
#define TPB 32
#define NUM_STREAMS 4

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len, int offset) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x; // local (segment index)
	const int globalId = i + offset;					 // global index
	if (globalId < len)
	{
		out[globalId] = in1[globalId] + in2[globalId];
	}
}


int main(int argc, char **argv) {
  
  int inputLength;
  int segmentSize;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ read in inputLength from args
  if (argc < 3) {
	  // make sure there is a number input
      std::cerr << "Usage: " << " <inputLength> <segLength> \n";
      return 1;
  }

  inputLength = std::atoi(argv[1]); // convert first argument from string to int
  segmentSize = std::atoi(argv[2]);
  printf("The input length is %d\n", inputLength);
  printf("The segment size is %d\n", segmentSize);
  
  //@@ Allocate Host memory for input and output
  hostInput1 = new DataType[inputLength];
  hostInput2 = new DataType[inputLength];
  hostOutput = new DataType[inputLength];
  resultRef = new DataType[inputLength];

  
  //@@ Initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
	  hostInput1[i] = static_cast<DataType>(rand()) / RAND_MAX;
	  hostInput2[i] = static_cast<DataType>(rand()) / RAND_MAX;
	  hostOutput[i] = 0;
	  resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Allocate GPU memory here
  size_t size = inputLength * sizeof(DataType);
  cudaMalloc(&deviceInput1, size);
  cudaMalloc(&deviceInput2, size);
  cudaMalloc(&deviceOutput, size);

  //@@ Create CUDA streams
  cudaStream_t streams[NUM_STREAMS];
  for (size_t i = 0; i < NUM_STREAMS; i++)
  {
	  cudaStreamCreate(&streams[i]);
  }

  struct timeval start, end;
  // Start the timer
  gettimeofday(&start, NULL);

  //@@ Launch the GPU Kernel in streams
  for (int i = 0; i < inputLength; i += segmentSize)
  {
	  // 1D grid and block dimensions
	  int size = min(segmentSize, inputLength - i);
	  int numBlocks = (size + TPB - 1) / TPB;

	  int streamId = (i / segmentSize) % NUM_STREAMS;
	  // Async copy from cpu to gpu
	  cudaMemcpyAsync(&deviceInput1[i], &hostInput1[i], size * sizeof(DataType), cudaMemcpyHostToDevice, streams[streamId]);
	  cudaMemcpyAsync(&deviceInput2[i], &hostInput2[i], size * sizeof(DataType), cudaMemcpyHostToDevice, streams[streamId]);

	  vecAdd <<<numBlocks, TPB, 0, streams[streamId]>>> (deviceInput1, deviceInput2, deviceOutput, inputLength, i);

	  // Async copy from gpu to cpu
	  cudaMemcpyAsync(&hostOutput[i], &deviceOutput[i], size * sizeof(DataType), cudaMemcpyDeviceToHost, streams[streamId]);
  }

  // Wait for all streams to complete
  for (int i = 0; i < NUM_STREAMS; i++) {
	  cudaStreamSynchronize(streams[i]);
  }

  // Check error
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
	  fprintf(stderr, "Kernel launch failed - %s\n", cudaGetErrorString(err));
  }

  // Stop the timer
  gettimeofday(&end, NULL);
  long seconds = end.tv_sec - start.tv_sec;
  long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
  std::cout << "Time taken in total of: " << micros << " microseconds" << std::endl;

  //@@ Insert code below to compare the output with the reference
  bool isCorrect = true;
  for (int i = 0; i < inputLength; i++) {
	  if (fabs(resultRef[i] - hostOutput[i]) > 1e-5) { // Use an epsilon value for floating-point comparison
		  isCorrect = false;
		  std::cout << "Result Ref" << i << " is " << resultRef[i] << " but device result is" << hostOutput[i] << std::endl;
		  break;
	  }
  }
  std::cout << "Results are " << (isCorrect ? "correct" : "incorrect") << std::endl;

  // Cleanup streams
  for (int i = 0; i < NUM_STREAMS; i++) {
	  cudaStreamDestroy(streams[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  delete[] hostInput1;
  delete[] hostInput2;
  delete[] hostOutput;
  delete[] resultRef;


  return 0;
}
