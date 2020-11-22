#include <stdio.h>

void checkCUDAError(const char *msg);

#include <stdio.h>

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*257)
#define FIXME 0
#define FIXME1 1
#define FIXME2 2

// The FIXME's indicate where code must be added to replace them.
// The number of output elements is N, out[0:N-1]
// The number of input elements is N+2*RADIUS, IN[0:N+2*RADIUS-1]
// Each element of out holds the sum of a set of 2*RADIUS+1 contiguous elements from in
// The sum of contents in in[0:2*RADIUS] is placed in out[0], 
// sum of elements in in[1:2*RADIUS+1] is placed in out[1], etc.

__global__ void stencil_1d(int *in, int *out, int N) 
{
  int gindex = blockDim.x * blockIdx.x + threadIdx.x;

  if (gindex >= RADIUS && (N + 2 * RADIUS) - RADIUS > gindex)
  {
     // Apply the stencil
     int result = 0;
     for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
         result += in[gindex + offset];
 
     // Store the result
     out[gindex - RADIUS] = result;
  }
}

int main()
{
  unsigned int i;
  int *h_in, *h_out;
  h_in = new int[NUM_ELEMENTS + 2 * RADIUS];
  h_out = new int[NUM_ELEMENTS];
  int *d_in, *d_out;

  // Initialize host data
  for( i = 0; i < (NUM_ELEMENTS + 2*RADIUS); ++i )
    h_in[i] = 1; // With a value of 1 and RADIUS of 3, all output values should be 7

  // Allocate space on the device
  cudaMalloc( &d_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int));
  cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int));
  checkCUDAError("cudaMalloc");

  // Copy input data to device
  cudaMemcpy( d_in, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy");

  dim3 dimBlock (BLOCK_SIZE);
  dim3 dimGrid (ceil ((NUM_ELEMENTS + 2*RADIUS) / (float) BLOCK_SIZE));

  stencil_1d<<< dimGrid, dimBlock >>> (d_in, d_out, NUM_ELEMENTS);
  checkCUDAError("Kernel Launch Error:");

  cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMalloc");

  // Verify every out value is 7
  for( i = 0; i < NUM_ELEMENTS; ++i )
    if (h_out[i] != 7)
    {
      printf("ERROR: Element h_out[%d] == %d != 7\n", i, h_out[i]);
      break;
    }

  if (i == NUM_ELEMENTS)
    printf("SUCCESS!\n");

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);
  delete h_in;
  delete h_out;

  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

