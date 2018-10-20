/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
int const NUM_STREAMS = 16;
int const THREAD_PER_BLOCK = 1024;

__global__
void predictAndScan(const unsigned int* const vals,
unsigned int *predictCollect,
unsigned int *scanCollect,
int binNum,
unsigned int numBins,
int numVals)
{
  __shared__ unsigned int s_predict[THREAD_PER_BLOCK];
  __shared__ unsigned int s_predictTMP[THREAD_PER_BLOCK];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int x = threadIdx.x;
  if (vals[idx] / NUM_STREAMS == binNum) s_predict[x] = 1;

  predictCollect[numVals * binNum + idx] = s_predict;

  int dist = 1;
  int count = 0;

  while (dist < THREAD_PER_BLOCK) {
    if (count % 2 == 0){
      s_predictTMP[x] = s_predict[x];
      if (x >= dist) {
        s_predictTMP[x] += s_predict[x - dist];
      }
    }
    else {
      s_predict[x] = s_predictTMP[x];
      if (x >= dist) {
        s_predict[x] += s_predictTMP[x - dist];
      }
    }
    dist *= 2;
    count++;
    __syncthreads(); 
  }
  if (count % 2 == 0) scanCollect[numVals * binNum + idx] = s_predict[x];
  else scanCollect[numVals * binNum + idx] = s_predictTMP[x];
}


__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

__global__
void baseline(const unsigned int* const vals, //INPUT
              unsigned int* const histo,      //OUPUT
              const unsigned int numBins,
              int numVals)
{
  __shared__ unsigned int s_histo[numBins];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int x = threadIdx.x;
  atomicAdd(&s_histo[vals[idx]], 1);

  __syncthreads();
  while(x < numBins){
    atomicAdd(&histo[x], s_histo[x]);
    x += THREAD_PER_BLOCK;
  }
}

__global__
void reduceOneByOne(const unsigned int* const vals, //INPUT
                    unsigned int *sums,
                    int binNum,
                    int numBins,
                    int numElems)
{
  __shared__ unsigned int s_sums[THREAD_PER_BLOCK];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= numElems) return;
  int x = threadIdx.x;
  int bx = blockIdx.x;
  s_sums[x] = (vals[idx] == binNum);
  __syncthreads();
  int half = THREAD_PER_BLOCK / 2;

  while(half != 0) {
    if(x < half) {
      s_sums[x] += s_sums[x + half];
    }
    half /= 2;
    __syncthreads();
  }
  if (x == 0) sums[bx] = s_sums[0];

}

__global__
void reduceOnly(const unsigned int* const vals, //INPUT
  unsigned int *sums,
  int numBins,
  int numElems)
{
  __shared__ unsigned int s_sums[THREAD_PER_BLOCK];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= numElems) return;
  int x = threadIdx.x;
  int bx = blockIdx.x;
  s_sums[x] = vals[idx];
  __syncthreads();
  int half = THREAD_PER_BLOCK / 2;

  while(half != 0) {
    if(x < half) {
      s_sums[x] += s_sums[x + half];
    }
    half /= 2;
    __syncthreads();
  }
  if (x == 0) sums[bx] = s_sums[0];

}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{

  /*
  Solution 1
  Just a atomics
  */

  int numBlock = (numElems + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  baseline<<<numBlock,THREAD_PER_BLOCK>>>(d_vals, //INPUT
    d_histo,      //OUPUT
    numBins,
    numElems);
  

  /*
  1. Compact into coarse bin
  2. Concurrently stream coarse bin into bins
  3. Concate all coarse
  */
  //TODO Launch the yourHisto kernel


  /*
  1. Iterate through numBins
  2. Predict and Sum
  This might be done in O(KlogN)
  */

  /*
  cudaStream_t streams[NUM_STREAMS];
  unsigned int *inter_sum[numBins]; 
  for (int i = 0; i < numBins; i++) {
    cudaStreamCreate(&streams[i]);
    int numBlock = numElems / THREAD_PER_BLOCK + 1; // Lazy
    cudaMalloc(&inter_sum[i], sizeof(int) * numBlock);
    reduceOneByOne<<<numBlock, THREAD_PER_BLOCK, streams[i]>>>(d_vals, inter_sum[i], 
      i, numBins, numElems);

    while(numBlock != 0) {
      int tmp_numBlock = numBlock / THREAD_PER_BLOCK + 1;
      reduceOnly<<<tmp_numBlock, THREAD_PER_BLOCK, streams[i]>>>(d_vals, inter_sum[i], 
        numBins, numElems);

      numBlock = tmp_numBlock;
    }

    cudaMemcpyAsync(&d_histo[i], &inter_sum[i][0], sizeof(int),
       cudaMemcpyDeviceToDevice, streams[i]);
  }
  */

  //if you want to use/launch more than one kernel,
  //feel free
}
