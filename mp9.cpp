// Excercise 9 - OpenACC : Vector Addition

#include <wb.h>

void vadd(const float * a, const float * b, float * result, int len) {

  #pragma acc parallel loop copyin(a[0:len]) copyin(b[0:len]) copyout(result[0:len])
  for (int i = 0; i < len; i++) {
    result[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Performing OpenACC computation");

  vadd(hostInput1, hostInput2, hostOutput, inputLength);

  wbTime_stop(GPU, "Performing OpenACC computation");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
