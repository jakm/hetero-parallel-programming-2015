// Excercise 2.1 - Basic Matrix-Matrix Multiplication

#include <wb.h>
#include <assert.h>

#define BLOCK_SIZE 16

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  int row = threadIdx.y+blockDim.y*blockIdx.y;
  int col = threadIdx.x+blockDim.x*blockIdx.x;

  if (row < numCRows && col < numCColumns) {
    float Cvalue = 0.0;
    for (int i = 0; i < numAColumns; i++) { // i < numBRows
      Cvalue += A[row*numAColumns+i] * B[numBColumns*i+col];
    }
    C[row*numCColumns+col] = Cvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns

  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  assert (numAColumns == numBRows);
  size_t matrixASize = numARows * numAColumns * sizeof(float);
  size_t matrixBSize = numBRows * numBColumns * sizeof(float);
  size_t matrixCSize = numARows * numBColumns * sizeof(float);

  hostC = ( float * )malloc(matrixCSize);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  wbCheck(cudaMalloc((void **) &deviceA, matrixASize));
  wbCheck(cudaMalloc((void **) &deviceB, matrixBSize));
  wbCheck(cudaMalloc((void **) &deviceC, matrixCSize));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  wbCheck(cudaMemcpy(deviceA, hostA, matrixASize, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, matrixBSize, cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  dim3 DimGrid((numBColumns-1)/BLOCK_SIZE+1, (numARows-1)/BLOCK_SIZE+1, 1);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows,
                                        numAColumns, numBRows, numBColumns,
                                        numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  wbCheck(cudaMemcpy(hostC, deviceC, matrixCSize, cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
