//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */


/*
Channel Separate from P.2
*/

#include "utils.h"
#include <thrust/host_vector.h>
#include <vector>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ( x < numRows && y < numCols) {
    int offset = x * numCols + y;
    uchar4 rgba = inputImageRGBA[offset];
    redChannel[offset] = (unsigned char)rgba.x;
    greenChannel[offset] = (unsigned char)rgba.y;
    blueChannel[offset] = (unsigned char)rgba.z;
  }
}

__global__ 
void sourceMask(const uchar4* const sourceImg,
                      unsigned char* sourceMask,
                      const size_t numRowsSource, 
                      const size_t numColsSource)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  int offset = x * numColsSource + y;
  if (x < numRowsSource && y < numColsSource){
    sourceMask[offset] = ((sourceImg[offset].x + sourceImg[offset].y 
      + sourceImg[offset].z) < 255 * 3) ? 1 : 0;
  }
}

__global__ 
void isStrictInterior(
  unsigned char* sourceMask,
  //unsigned char* strictInteriorPixels,
  unsigned char* borderPixels,
  unsigned char* interiorPixels,
  const size_t numRowsSource, 
  const size_t numColsSource
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  if (x >= numRowsSource || y >= numColsSource) return;
  int offset = x * numColsSource + y;
  //if (offset >= numColsSource * numRowsSource) return;
  if (!sourceMask[offset]) {
    borderPixels[offset] = 0;
    interiorPixels[offset] = 0;
    //strictInteriorPixels[offset] = 0;
  }
  else if (sourceMask[(x - 1) * numColsSource + y] && sourceMask[(x + 1) * numColsSource + y]
    && sourceMask[x * numColsSource + y - 1] && sourceMask[x * numColsSource + y + 1]){
      //strictInteriorPixels[offset] = 1;
      interiorPixels[offset] = 1;
      borderPixels[offset] = 0;
    }
  else {
    //strictInteriorPixels[offset] = 0;
    interiorPixels[offset] = 0;
    borderPixels[offset] = 1;
  }
}

__global__ 
void debugMask(
  unsigned char* sourceMask,
  uchar4* d_out,
  const size_t numRowsSource, 
  const size_t numColsSource
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  if (x >= numRowsSource || y >= numColsSource) return;
  int offset = x * numColsSource + y;

  //if (offset >= (numRowsSource * numColsSource)) return;
  if (sourceMask[offset]){
    d_out[offset].x = 255;
    d_out[offset].y = 255;
    d_out[offset].z = 255;
    d_out[offset].w= 255;
  }
  else {
    d_out[offset].x = 0;
    d_out[offset].y = 0;
    d_out[offset].z = 0;
    d_out[offset].w= 255;
  }
}

__global__ 
void debugBorder(
  unsigned char* sourceMask,
  unsigned char* borderPixels,
  unsigned char* interiorPixels,
  uchar4* d_out,
  const size_t numRowsSource, 
  const size_t numColsSource
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  if (x >= numRowsSource || y >= numColsSource) return;
  int offset = x * numColsSource + y;

  //if (offset >= (numRowsSource * numColsSource)) return;
  if (borderPixels[offset]){
    d_out[offset].x = 220;
    d_out[offset].y = 20;
    d_out[offset].z = 60;
    d_out[offset].w= 255;
  }
  else if (interiorPixels[offset]){
    d_out[offset].x = 0;
    d_out[offset].y = 128;
    d_out[offset].z = 0;
    d_out[offset].w= 255;
  }
  else {
    d_out[offset].x = 0;
    d_out[offset].y = 0;
    d_out[offset].z = 0;
    d_out[offset].w= 255;
  }
}

__global__
void copy(
  float* to_this,
  unsigned char* from,
  const size_t numRowsSource, 
  const size_t numColsSource
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  int offset = x * numColsSource + y;
  if (x >= numRowsSource || y >= numColsSource) return;
  //if (offset >= numColsSource * numRowsSource) return;
  to_this[offset] = (float)from[offset];

}

__global__
void computeG(
  unsigned char* channel,
  float* g,
  unsigned char* d_interiorPixels,
  const size_t numRowsSource, 
  const size_t numColsSource
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  if (x >= numRowsSource || y >= numColsSource) return;
  int offset = x * numColsSource + y;
  //if (offset >= numRowsSource * numColsSource) return;
  if (d_interiorPixels[offset]) {
    float sum = 4.f * channel[offset];
    sum -= (float)channel[offset - numColsSource] + (float)channel[offset + numColsSource];
    sum -= (float)channel[offset - 1] + (float)channel[offset + 1];
    g[offset] = sum;
  } else {
    g[offset] = 0;
  }
}

__global__
void naiveJacobi(
  const unsigned char* const d_destImg,
  const unsigned char* const d_interiorPixels,
  const unsigned char* const d_borderPixels,
  const size_t numColsSource,
  const size_t numRowsSource,
  float* const f,
  const float* const g,
  float* const f_next
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  if (x >= numRowsSource || y >= numColsSource) return;
  int offset = x * numColsSource + y;
  float sum = 0.f;
  int iterate[4] = {offset - 1, offset + 1, (int)(offset - numColsSource), (int)(offset + numColsSource)};
  if (!d_interiorPixels[offset]) return;
  for (int i = 0; i < 4; i++) {
    int coor = iterate[i];
    sum += d_interiorPixels[coor] * f[coor] + (1 - d_interiorPixels[coor]) * d_destImg[coor];
  }
  float f_next_val = (sum + g[offset]) / 4.f;
  f_next_val = min(255.f, max(0.f, f_next_val));
  f_next[offset] = f_next_val;
}

__global__
void pasteImage(
  uchar4* d_destImg,
  const unsigned char* const d_interiorPixels,
  float* blendedValsRed,
  float* blendedValsGreen,
  float* blendedValsBlue,
  const size_t numColsSource,
  const size_t numRowsSource
)
{
  int idx = threadIdx.x, idy = threadIdx.y, bdx = blockIdx.x, bdy = blockIdx.y;
  int dimx = blockDim.x, dimy = blockDim.y;
  int x = bdx * dimx + idx;
  int y = bdy * dimy + idy;
  if (x >= numRowsSource || y >= numColsSource) return;
  int offset = x * numColsSource + y;
  //if (offset >= numColsSource * numRowsSource) return;
  if (!d_interiorPixels[offset]) return;

  d_destImg[offset].x = (unsigned char)blendedValsRed[offset];
  d_destImg[offset].y = (unsigned char)blendedValsGreen[offset];
  d_destImg[offset].z = (unsigned char)blendedValsBlue[offset];
  
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  const unsigned int KERNEL_DIM = 16;
  unsigned int numPixel = numColsSource * numRowsSource;
  unsigned char* d_sourceMask;
  //unsigned char* d_strictInteriorPixels;
  unsigned char* d_borderPixels;
  unsigned char* d_interiorPixels;
  uchar4* d_sourceImg;
  uchar4* d_destImg;
  uchar4* d_blendedImg;
  checkCudaErrors(cudaMallocManaged(&d_sourceImg, sizeof(uchar4) * numPixel));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numPixel, 
    cudaMemcpyHostToDevice));
  
  checkCudaErrors(cudaMallocManaged(&d_destImg, sizeof(uchar4) * numPixel));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * numPixel, 
    cudaMemcpyHostToDevice));
  
  checkCudaErrors(cudaMallocManaged(&d_blendedImg, sizeof(uchar4) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_sourceMask, sizeof(unsigned char) * numPixel));
  //checkCudaErrors(cudaMallocManaged(&d_strictInteriorPixels, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_borderPixels, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_interiorPixels, sizeof(unsigned char) * numPixel));


  //Step 1
  int rowBlock = (numRowsSource + KERNEL_DIM - 1)/KERNEL_DIM; 
  int colBlock = (numColsSource + KERNEL_DIM - 1)/KERNEL_DIM; 
  const dim3 blockSize(rowBlock, colBlock, 1);
  const dim3 kernelSize(KERNEL_DIM, KERNEL_DIM, 1);
  sourceMask<<<blockSize, kernelSize>>>(
    d_sourceImg,
    d_sourceMask,
    numRowsSource, 
    numColsSource
  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // debugMask<<<blockSize, kernelSize>>>(
  //   d_sourceMask,
  //   d_blendedImg,
  //   numRowsSource, 
  //   numColsSource
  // );
  // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  // checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * numPixel, 
  //   cudaMemcpyDeviceToHost));
  

  //Step 2

  isStrictInterior<<<blockSize, kernelSize>>>(
    d_sourceMask,
    d_borderPixels,
    d_interiorPixels,
    numRowsSource, 
    numColsSource
  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  debugBorder<<<blockSize, kernelSize>>>(
    d_sourceMask,
    d_borderPixels,
    d_interiorPixels,
    d_blendedImg,
    numRowsSource, 
    numColsSource
  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * numPixel, 
    cudaMemcpyDeviceToHost));

  //return;

  // Step 3

  unsigned char* d_srcRedChannel;
  unsigned char* d_srcGreenChannel;
  unsigned char* d_srcBlueChannel;

  unsigned char* d_destRedChannel;
  unsigned char* d_destGreenChannel;
  unsigned char* d_destBlueChannel;

  checkCudaErrors(cudaMallocManaged(&d_srcRedChannel, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_srcGreenChannel, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_srcBlueChannel, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_destRedChannel, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_destGreenChannel, sizeof(unsigned char) * numPixel));
  checkCudaErrors(cudaMallocManaged(&d_destBlueChannel, sizeof(unsigned char) * numPixel));

  separateChannels<<<blockSize, kernelSize>>>(
    d_sourceImg,
    numRowsSource,
    numColsSource,
    d_srcRedChannel,
    d_srcGreenChannel,
    d_srcBlueChannel
  );

  separateChannels<<<blockSize, kernelSize>>>(
    d_destImg,
    numRowsSource,
    numColsSource,
    d_destRedChannel,
    d_destGreenChannel,
    d_destBlueChannel
  );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Step 4

  float *blendedValsRed_1;
  float *blendedValsRed_2;
  float *blendedValsGreen_1;
  float *blendedValsGreen_2;
  float *blendedValsBlue_1;
  float *blendedValsBlue_2;
  float *g_red;
  float *g_green;
  float *g_blue;

  checkCudaErrors(cudaMallocManaged(&blendedValsRed_1, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&blendedValsRed_2, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&blendedValsGreen_1, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&blendedValsGreen_2, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&blendedValsBlue_1, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&blendedValsBlue_2, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&g_blue, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&g_green, sizeof(float) * numPixel));
  checkCudaErrors(cudaMallocManaged(&g_red, sizeof(float) * numPixel));

  copy<<<blockSize, kernelSize>>>(
    blendedValsRed_1, d_srcRedChannel, numRowsSource, numColsSource
  );
  copy<<<blockSize, kernelSize>>>(
    blendedValsRed_2, d_srcRedChannel, numRowsSource, numColsSource
  );
  copy<<<blockSize, kernelSize>>>(
    blendedValsGreen_1, d_srcGreenChannel, numRowsSource, numColsSource
  );
  copy<<<blockSize, kernelSize>>>(
    blendedValsGreen_2, d_srcGreenChannel, numRowsSource, numColsSource
  );
  copy<<<blockSize, kernelSize>>>(
    blendedValsBlue_1, d_srcBlueChannel, numRowsSource, numColsSource
  );
  copy<<<blockSize, kernelSize>>>(
    blendedValsBlue_2, d_srcBlueChannel, numRowsSource, numColsSource
  );
  

  // checkCudaErrors(cudaMemcpy(blendedValsRed_1, d_srcRedChannel, sizeof(float) * numPixel,
  //   cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(blendedValsRed_2, d_srcRedChannel, sizeof(float) * numPixel,
  //   cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(blendedValsGreen_1, d_srcGreenChannel, sizeof(float) * numPixel,
  //   cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(blendedValsGreen_2, d_srcGreenChannel, sizeof(float) * numPixel,
  //   cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(blendedValsBlue_1, d_srcBlueChannel, sizeof(float) * numPixel,
  //   cudaMemcpyDeviceToDevice));
  // checkCudaErrors(cudaMemcpy(blendedValsBlue_2, d_srcBlueChannel, sizeof(float) * numPixel,
  //   cudaMemcpyDeviceToDevice));

  computeG<<<blockSize, kernelSize>>>( d_srcRedChannel, g_red, 
    d_interiorPixels, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  computeG<<<blockSize, kernelSize>>>( d_srcBlueChannel, g_blue, 
    d_interiorPixels, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  computeG<<<blockSize, kernelSize>>>( d_srcGreenChannel, g_green, 
    d_interiorPixels, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Step 5

  for (int i = 0; i < 800; ++i) {
    naiveJacobi <<<blockSize, kernelSize>>>(
      d_destRedChannel,
      d_interiorPixels,
      d_borderPixels,
      numColsSource,
      numRowsSource,
      blendedValsRed_1,
      g_red,
      blendedValsRed_2
    );

    naiveJacobi <<<blockSize, kernelSize>>>(
      d_destGreenChannel,
      d_interiorPixels,
      d_borderPixels,
      numColsSource,
      numRowsSource,
      blendedValsGreen_1,
      g_green,
      blendedValsGreen_2
    );

    naiveJacobi <<<blockSize, kernelSize>>>(
      d_destBlueChannel,
      d_interiorPixels,
      d_borderPixels,
      numColsSource,
      numRowsSource,
      blendedValsBlue_1,
      g_blue,
      blendedValsBlue_2
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(blendedValsRed_1, blendedValsRed_2, sizeof(float) * numPixel,
    cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(blendedValsGreen_1, blendedValsGreen_2, sizeof(float) * numPixel,
    cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(blendedValsBlue_1, blendedValsBlue_2, sizeof(float) * numPixel,
    cudaMemcpyDeviceToDevice));
    
  }

  pasteImage<<<blockSize, kernelSize>>>(
    d_destImg,
    d_interiorPixels,
    blendedValsRed_1,
    blendedValsGreen_1,
    blendedValsBlue_1,
    numColsSource,
    numRowsSource
  );

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_destImg, sizeof(uchar4) * numPixel,
  cudaMemcpyDeviceToHost));
  

  //Free memory
  checkCudaErrors(cudaFree(d_sourceImg));
  checkCudaErrors(cudaFree(d_destImg));
  checkCudaErrors(cudaFree(d_sourceMask));
  checkCudaErrors(cudaFree(d_blendedImg));
  checkCudaErrors(cudaFree(d_borderPixels));
  checkCudaErrors(cudaFree(d_interiorPixels));

  checkCudaErrors(cudaFree(blendedValsRed_1));
  checkCudaErrors(cudaFree(blendedValsRed_2));
  checkCudaErrors(cudaFree(blendedValsGreen_1));
  checkCudaErrors(cudaFree(blendedValsGreen_2));
  checkCudaErrors(cudaFree(blendedValsBlue_1));
  checkCudaErrors(cudaFree(blendedValsBlue_2));

  checkCudaErrors(cudaFree(g_blue));
  checkCudaErrors(cudaFree(g_green));
  checkCudaErrors(cudaFree(g_red));
  
  checkCudaErrors(cudaFree(d_srcRedChannel));
  checkCudaErrors(cudaFree(d_srcGreenChannel));
  checkCudaErrors(cudaFree(d_srcBlueChannel));

  checkCudaErrors(cudaFree(d_destRedChannel));
  checkCudaErrors(cudaFree(d_destGreenChannel));
  checkCudaErrors(cudaFree(d_destBlueChannel));

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
}
