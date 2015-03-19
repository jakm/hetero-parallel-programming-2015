// Excercise 7 - Vector Addition with Streams

/******************************************************************************
 *                                                                            *
 * TODO: streamy funguji pouze s pinned pameti (cudaMallocHost); nejak overit,*
 * zda bude segmentovani fungovat tak, jak je ted                             *
 *                                                                            *
 ******************************************************************************/

#include <wb.h>

#define wbCheck(stmt) do {                                                \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)

#define BLOCK_SIZE 256
#define SEGMENT_SIZE 1024

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here

    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    int noSegments = ((inputLength-1)/SEGMENT_SIZE) + 1;

    wbLog(TRACE, "Number of streams is ", noSegments);

    cudaStream_t * streams = (cudaStream_t *) malloc(noSegments * sizeof(cudaStream_t));
    if (streams == NULL) {
        wbLog(ERROR, "Memory allocation failed - array of streams");
        return -1;
    }

    for (int i = 0; i < noSegments; i++) {
        wbCheck(cudaStreamCreate(&streams[i]));
    }

    int inputSize = inputLength * sizeof(float);

    wbTime_start(GPU, "Allocating GPU memory.");

    wbCheck(cudaMalloc((void **) &deviceInput1, inputSize));
    wbCheck(cudaMalloc((void **) &deviceInput2, inputSize));
    wbCheck(cudaMalloc((void **) &deviceOutput, inputSize));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(Compute, "Performing CUDA computation");

    dim3 dimGrid((SEGMENT_SIZE-1/BLOCK_SIZE)+1);
    dim3 dimBlock(BLOCK_SIZE);

    int offset;
    int idxWithPartialCopy;

    if (inputLength % SEGMENT_SIZE == 0) { // data are aligned to block size
        idxWithPartialCopy = -1;
    } else {
        idxWithPartialCopy = noSegments - 1;
    }

    int segmentSize = SEGMENT_SIZE * sizeof(float);

    for (int i = 0; i < noSegments; i++) {
        offset = i * SEGMENT_SIZE;

        if (i == idxWithPartialCopy) {
            segmentSize = (inputLength % SEGMENT_SIZE) * sizeof(float);
        }

        wbCheck(cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], segmentSize, cudaMemcpyHostToDevice, streams[i]));
        wbCheck(cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], segmentSize, cudaMemcpyHostToDevice, streams[i]));

        vecAdd<<<dimGrid, dimBlock, 0, streams[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], inputLength);

        wbCheck(cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], segmentSize, cudaMemcpyDeviceToHost, streams[i]));
    }

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation");

    for (int i = 0; i < noSegments; i++) {
        cudaStreamDestroy(streams[i]);
    }

    wbTime_start(GPU, "Freeing GPU Memory");

    wbCheck(cudaFree(deviceOutput));
    wbCheck(cudaFree(deviceInput2));
    wbCheck(cudaFree(deviceInput1));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

