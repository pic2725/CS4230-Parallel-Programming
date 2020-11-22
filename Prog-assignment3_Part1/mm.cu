#include <stdio.h>
#include <time.h>

#define FIXME0 0
#define FIXME1 1
#define FIXME2 2

void checkCUDAError(const char *msg);

const int DSIZE = 6001;
const float A_val = 3.0f;
const float B_val = 2.0f;
cudaEvent_t start, stop;
float elapsedTime;

// matrix multiply kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  int idx = threadIdx.x + (blockIdx.x * blockDim.x); // create thread x index
  int idy = threadIdx.y + (blockIdx.y * blockDim.y); // create thread y index

  if ((idx < ds) && (idy < ds)){
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[(idx * ds) + i] * B[(i * ds) + idy];   // dot product of row and column
    C[idx * ds + idy] = temp;
  }
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // start timing

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;}


  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D failure");

  dim3 block(1,1);  
  dim3 grid(1,1);
  int Bx, By;
  printf("Matrix size: %d\n", DSIZE);
  while(1)
 {
  printf("Specify TB-size-x,TB-size-y: ");
  scanf("%d %d", &Bx,&By);
  if ((Bx==0) or (By==0)) break;
  block.x = Bx;
  block.y = By;
  grid.x = ceil(DSIZE / (float) block.x);
  grid.y = ceil(DSIZE / (float) block.y);

  for(int trial=0;trial<5;trial++)
  {
  cudaEventCreate(&start);
  cudaEventRecord(start,0);
   // Launch kernel
   mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
   checkCUDAError("kernel launch failure");
   cudaEventCreate(&stop);
   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsedTime, start,stop);
   cudaDeviceSynchronize();
   printf("Trial %d: GFLOPS: %.2f\n",trial,2.0e-6*DSIZE*DSIZE*DSIZE/elapsedTime);
   // Copy results back to host
   cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy D2H failure");
   for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("Error: mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
  }
 }
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

