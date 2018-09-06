/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <float.h>

const int block_size = 1024;
const int DIM = 32;
const int MAX_MEM = 10000;

#ifndef max

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#endif



#ifndef min

#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#endif


__global__ void calculate_maxmin(const float* const d_logLuminance,
  float *blockCollectMax,
  float *blockCollectMin,
  const size_t numRows,
  const size_t numCols,
  int numBlockx,
  int numBlocky)                              
{
  __shared__ float sluminance_max[block_size];
  __shared__ float sluminance_min[block_size];
  int idx = threadIdx.x, idy = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  int offset = x + y * numCols;
  int idx_offset = idx + idy * DIM;
  //if (blockIdx.x == 2 && blockIdx.y == 2)
  //printf("%f ", d_logLuminance[offset]);

  if (x < numCols && y < numRows) {
    sluminance_max[idx_offset] = d_logLuminance[offset];
    sluminance_min[idx_offset] = d_logLuminance[offset];
  } else {
    sluminance_max[idx_offset] = 0.f;
    sluminance_min[idx_offset] = FLT_MAX;
  }

  __syncthreads();

  int half = block_size / 2;
  while (half != 0) {
    if (idx_offset < half){
      sluminance_max[idx_offset] = max(sluminance_max[idx_offset], sluminance_max[idx_offset + half]);
      sluminance_min[idx_offset] = min(sluminance_min[idx_offset], sluminance_min[idx_offset + half]);
    }
    half /= 2;
    __syncthreads();
  }

  blockCollectMax[blockIdx.y * numBlockx + blockIdx.x] = sluminance_max[0];
  blockCollectMin[blockIdx.y * numBlockx + blockIdx.x] = sluminance_min[0];

  /*__syncthreads();

  int N = numBlockx * numBlocky;
  int res_idx = blockIdx.y * numBlockx + blockIdx.x;
  half = N / 2;
  while (half != 0) {
    if (res_idx < half){
      blockCollectMax[res_idx] = max(blockCollectMax[res_idx], blockCollectMax[res_idx + half]);
      blockCollectMin[res_idx] = min(blockCollectMin[res_idx], blockCollectMin[res_idx + half]);
    }
    half /= 2;
    __syncthreads();
  }*/


}

__global__ void collect_histo(const float* const d_logLuminance,
                              unsigned int *histo,
                              unsigned int *collects,
                              float logLumRange,
                              float min_logLum,
                              const size_t numRows,
                              const size_t numCols,
                              const size_t numBins)
{
  int idx = threadIdx.x, idy = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  if (x >= numCols && y >= numRows) return;
  int offset = x + y * numCols;
  int idx_offset = idx + idy * DIM;

  int bin_block = min((d_logLuminance[offset] - min_logLum) * numBins / logLumRange,
                       numBins - 1); 

  atomicAdd(&collects[bin_block], 1);
  //collects[offset * numBins + bin_block] = 1;


  /*__syncthreads();

  int N = numCols * numRows;
  int half = N / 2;

  while (half != 0) {
    if(offset < half){
      for (int i = 0; i < numBins; i++){ 
	collects[offset * numBins + i] += collects[(half + offset) * numBins + i];
      }
    }
    __syncthreads();
    half /= 2;
  }
*/
}

__global__ void cdf_count(unsigned int *collects,
unsigned int* const d_cdf,
const size_t numBins)
{
  int idx = threadIdx.x;
  if (idx >= numBins) return;
  __shared__ unsigned int smemory[MAX_MEM];
  // Implement Hillis / Steele
  smemory[idx] = collects[idx];
  printf("%d ", collects[idx]);
  int dist = 1;
  __syncthreads();

  while (dist < numBins) {
    if ((idx - dist) >= 0) {
      smemory[idx] += smemory[idx - dist];
    }
    dist *= 2;
    __syncthreads();
  }
  __syncthreads();
  d_cdf[idx] = smemory[idx];
 
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // Step 1
  const dim3 blockSize(DIM, DIM, 1);
  const dim3 gridSize((numCols + DIM - 1) / DIM, (numRows + DIM - 1) / DIM, 1);
  int numBlockx = (numCols + DIM - 1) / DIM;
  int numBlocky = (numRows + DIM - 1) / DIM;
  float *blockCollectMax, *blockCollectMin;
  checkCudaErrors(cudaMallocManaged(&blockCollectMax, sizeof(float) * numBlockx * numBlocky));
  checkCudaErrors(cudaMallocManaged(&blockCollectMin, sizeof(float) * numBlockx * numBlocky));

  calculate_maxmin<<<gridSize,blockSize>>>(d_logLuminance,
                                        blockCollectMax,
                                        blockCollectMin,
                                        numRows,
                                        numCols,
                                        numBlockx,
                                        numBlocky);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  max_logLum = blockCollectMax[0];                                      
  min_logLum = blockCollectMin[0];

  for(int i = 0; i < numBlockx * numBlockx; i++ ) {
    max_logLum = max(max_logLum, blockCollectMax[i]);
    min_logLum = min(min_logLum, blockCollectMin[i]);   
  }  
  printf("%f %f", max_logLum, min_logLum);
  // Step 2 Diff
  float logLumRange = max_logLum - min_logLum;  

  // Step 3 Histrogram
  unsigned int *histo;
  unsigned int *collects;
  checkCudaErrors(cudaMallocManaged(&histo, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMallocManaged(&collects, sizeof(unsigned int) * numBins));
  //checkCudaErrors(cudaMallocManaged(&collects, sizeof(unsigned int) * numBins * numCols * numRows));

  collect_histo<<<gridSize,blockSize>>>(d_logLuminance,
                                        histo,
                                        collects,
                                        logLumRange,
                                        min_logLum,
                                        numRows,
                                        numCols,
                                        numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
  // Step 4 CDF
  cdf_count<<<1, numBins>>>(collects, d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
  
  checkCudaErrors(cudaFree(blockCollectMax));
  checkCudaErrors(cudaFree(blockCollectMin));
  checkCudaErrors(cudaFree(histo));
  checkCudaErrors(cudaFree(collects));
}



