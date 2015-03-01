// Excercise 5 - List Scan

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}


/******************************************************************************
 * FIXME: nefunguje spravne pro dataset 8                                     *
 * TODO: implementovat verzi s polovicnim poctem vlaken, generizovat druhe    *
 *       volani scan, vyhodnit nadbytecne volani cudaDeviceSynchronize()      *
 ******************************************************************************/


#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    __shared__ float XY[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    XY[threadIdx.x] = i < len ? input[i] : 0.0;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (threadIdx.x+1) * stride * 2 - 1;
        if (index < blockDim.x)
            XY[index] += XY[index-stride];
        __syncthreads();
    }

    for (int stride = blockDim.x/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1) * stride * 2 - 1;
        if (index + stride < blockDim.x)
            XY[index+stride] += XY[index];
    }

    __syncthreads();

    if (i < len)
        output[i] = XY[threadIdx.x];
}

__global__ void vecAdd(float *vec, float *scalar, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
    vec[i] += *scalar;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    // dim3 DimGrid((numElements-1)/(BLOCK_SIZE<<1) + 1, 1, 1);
    dim3 DimGrid((numElements-1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);

    wbCheck(cudaDeviceSynchronize());

    float * tmpSumsIn;
    float * tmpSumsOut;
    size_t tmpSumsElements = DimGrid.x;
    wbCheck(cudaMalloc((void**)&tmpSumsIn, tmpSumsElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&tmpSumsOut, tmpSumsElements*sizeof(float)));
    // wbCheck(cudaMemset(tmpSumsOut, 0, tmpSumsElements*sizeof(float)));

    for (int i = 0; i < tmpSumsElements; i++) {
        size_t lastElementInBlock = (i+1) * BLOCK_SIZE - 1;
        if (lastElementInBlock > numElements - 1) {
            lastElementInBlock = numElements - 1;
        }

        wbCheck(cudaMemcpy(&tmpSumsIn[i], &deviceOutput[lastElementInBlock], sizeof(float), cudaMemcpyDeviceToDevice));
    }

    wbCheck(cudaDeviceSynchronize());

    scan<<<1, BLOCK_SIZE>>>(tmpSumsIn, tmpSumsOut, tmpSumsElements);

    wbCheck(cudaDeviceSynchronize());

    for (int i = 0; i < tmpSumsElements - 1; i++) {
        vecAdd<<<1, BLOCK_SIZE>>>(&deviceOutput[(i+1)*BLOCK_SIZE], &tmpSumsOut[i], BLOCK_SIZE);
        wbCheck(cudaDeviceSynchronize());
    }

    float * xxx = (float *) malloc(tmpSumsElements*sizeof(float));
    wbCheck(cudaMemcpy(xxx, tmpSumsOut, tmpSumsElements*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < tmpSumsElements; i++) {
        printf("prefix %d (offset %d): %f\n", i, i*BLOCK_SIZE, xxx[i]);
    }
    free(xxx);


    wbCheck(cudaFree(tmpSumsIn));
    wbCheck(cudaFree(tmpSumsOut));

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
