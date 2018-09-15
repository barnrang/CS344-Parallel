//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <math.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

const int block_size = 1024;
const int DIM = 32;
const int MAX_THREADS_PER_BLOCK = 65535;
const int FIND_MAX_THREADS = 1024; //allocate to shared memory

__global__ void  findMax(unsigned int* const d_inputVals,
unsigned int *d_collectMax,
const size_t numElems)
{
  __shared__ unsigned int s_inputVals[FIND_MAX_THREADS];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < numElems) s_inputVals[threadIdx.x] = d_inputVals[idx];
  else s_inputVals[threadIdx.x] = 0;
  __syncthreads();

  int half = FIND_MAX_THREADS / 2;
  while (half != 0) {
    if (threadIdx.x < half) {
      s_inputVals[threadIdx.x] = max(s_inputVals[threadIdx.x], s_inputVals[threadIdx.x + half]);
    }
    half /= 2;
    __syncthreads();
  }
  d_collectMax[blockIdx.x] = s_inputVals[0];

}

__global__ void  scanSB(unsigned int* const d_inputVals, 
  unsigned int *d_collectScan,
  unsigned int *d_collectSumScan,
  unsigned int *d_sumBlock,
  unsigned int pos,
  size_t const numElems,
  unsigned int compare,
  int numMaxBlock) 
{
  __shared__ unsigned int s_inputVals[FIND_MAX_THREADS];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < numElems){
    s_inputVals[threadIdx.x] = (d_inputVals[idx] & pos) == compare;
    d_collectScan[idx] = s_inputVals[threadIdx.x];
  } else s_inputVals[threadIdx.x] = 0;
  __syncthreads();

  int dist = 1;

  while (dist < FIND_MAX_THREADS) {
    if (threadIdx.x >= dist) {
      s_inputVals[threadIdx.x] += s_inputVals[threadIdx.x - dist];
    }
    dist *= 2;
    __syncthreads();
  }
  d_collectSumScan[idx] = s_inputVals[threadIdx.x];
  d_sumBlock[blockIdx.x] = s_inputVals[FIND_MAX_THREADS - 1];
}

__global__ void  reduceBlockSum(unsigned int *d_sumBlock,
const size_t numMaxBlock)
{
  __shared__ unsigned int s_sumBlock[FIND_MAX_THREADS];
  int idx = threadIdx.x;
  if(idx >= numMaxBlock) return;
  s_sumBlock[idx] = d_sumBlock[idx];
  __syncthreads();

  int dist = 1;
  while (dist < numMaxBlock) {
    if (idx >= dist) {
      s_sumBlock[idx] += s_sumBlock[idx - dist];
    }
    dist *= 2;
    __syncthreads();
  }
  if(idx < numMaxBlock) d_sumBlock[idx + 1] = s_sumBlock[idx];
  else d_sumBlock[0] = 0;
}

__global__ void  mergeScan(unsigned int* const d_inputVals,
unsigned int* const d_inputPos,
unsigned int *d_collectScan,
unsigned int *d_collectSumScan,
unsigned int *d_sumBlock,
unsigned int *d_interVals,
unsigned int *d_interPos,
unsigned int offset)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (d_collectScan[idx]==0) return;
  d_interVals[d_collectSumScan[idx] + d_sumBlock[blockIdx.x] + offset - 1] = d_inputVals[idx];
  d_interPos[d_collectSumScan[idx] + d_sumBlock[blockIdx.x] + offset - 1] = d_inputPos[idx];
}

__global__ void  copyData(unsigned int* const d_inputVals, 
  unsigned int *d_interVals, 
  size_t const numElems)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= numElems) return;
  d_inputVals[idx] = d_interVals[idx];
}


void print_debug(unsigned int* const d_inputVals,
		unsigned int *d_interVals) {
  printf("Cut");
  for(int i = 0; i < 10; i++) {
    printf("%d ", d_inputVals[i]);
  }
  printf("\n");
  for(int i = 0; i < 10; i++) {
    printf("%d ", d_interVals[i]);
  }
  printf("\n");
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO

  // P.1 search for maximum
  printf("hello");
  unsigned int *d_collectMax;
  int numMaxBlock = (numElems + FIND_MAX_THREADS - 1)/FIND_MAX_THREADS;
  checkCudaErrors(cudaMallocManaged(&d_collectMax, sizeof(unsigned int) * numMaxBlock));
  printf("\n %d %d", numMaxBlock, FIND_MAX_THREADS);
  findMax <<<numMaxBlock,FIND_MAX_THREADS>>>(d_inputVals, d_collectMax, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  findMax <<<1, numMaxBlock>>>(d_collectMax, d_collectMax, numMaxBlock);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  unsigned int MAX = d_collectMax[0];
  printf("max %d \n", MAX);
  checkCudaErrors(cudaFree(d_collectMax));

  // P.2 Scan and Compact
  int N = log2(MAX);
  unsigned int MSB = 1;
  unsigned int *d_collectSumScan, *d_interVals, *d_interPos, *d_sumBlock;
  unsigned int *d_collectScan;
  unsigned int h_inputVals[numElems];
  checkCudaErrors(cudaMallocManaged(&d_collectSumScan, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMallocManaged(&d_collectScan, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMallocManaged(&d_interVals, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMallocManaged(&d_interPos, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMallocManaged(&d_sumBlock, sizeof(unsigned int) * (numMaxBlock+1)));
  for (int i = 0; i < N; ++i) {
    /*
    1. Predict & Scan through each block
    2. Reduce sum for each block
    3. compact elements by merging all block
    */
    scanSB<<<numMaxBlock,FIND_MAX_THREADS>>>(d_inputVals, 
      d_collectScan,
      d_collectSumScan,
      d_sumBlock,
      MSB,
      numElems,
      0, 
      numMaxBlock);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    reduceBlockSum<<<1,numMaxBlock>>>(d_sumBlock, numMaxBlock);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    mergeScan<<<numMaxBlock, FIND_MAX_THREADS>>>(d_inputVals,
      d_inputPos,
      d_collectScan,
      d_collectSumScan,
      d_sumBlock,
      d_interVals,
      d_interPos,
      0);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //printf("%d\n", d_collectScan[2]);
    //printf("%d\n", d_inputVals[2]);
    int offset = d_sumBlock[numMaxBlock];
    printf("\n %d", offset);
    scanSB<<<numMaxBlock,FIND_MAX_THREADS>>>(d_inputVals, 
      d_collectScan, 
      d_collectSumScan, 
      d_sumBlock, 
      MSB, 
      numElems, 
      MSB, 
      numMaxBlock);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    reduceBlockSum<<<1,numMaxBlock>>>(d_sumBlock, numMaxBlock);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    mergeScan<<<numMaxBlock, FIND_MAX_THREADS>>>(d_inputVals,
      d_inputPos,
      d_collectScan,
      d_collectSumScan,
      d_sumBlock,
      d_interVals,
      d_interPos,
      offset);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
      printf("Cut");
  for(int i = 0; i < 10; i++) {
    printf("%d ", h_inputVals[i]);
  }
  printf("\n");
  for(int i = 0; i < 10; i++) {
    printf("%d ", d_interVals[i]);
  }
  printf("%d ", d_interVals[numElems - 1]);
  printf("\n");
      //printf("%d \n", d_interVals[numElems - 2]);

     //print_debug(d_inputVals, d_interVals);
      
    checkCudaErrors(cudaMemcpy(d_inputPos, d_interPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputVals, d_interVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    MSB *= 2;
  }

  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  //PUT YOUR SORT HERE
  checkCudaErrors(cudaFree(d_collectSumScan));
  checkCudaErrors(cudaFree(d_collectScan));
  checkCudaErrors(cudaFree(d_interVals));
  checkCudaErrors(cudaFree(d_interPos));
  checkCudaErrors(cudaFree(d_sumBlock));
}
