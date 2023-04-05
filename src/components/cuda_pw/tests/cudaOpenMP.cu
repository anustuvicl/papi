/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Multi-GPU sample using OpenMP for threading on the CPU side
 * needs a compiler that supports OpenMP 2.0
 */

#include <papi.h>
#include "gpu_work.h"
#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe

#define PAPI_CALL(apiFuncCall)                                          \
do {                                                                           \
    int _status = apiFuncCall;                                         \
    if (_status != PAPI_OK) {                                              \
        fprintf(stderr, "%s:%d: error %d: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, _status, #apiFuncCall, PAPI_strerror(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define NUM_THREADS 8
// User metrics to profile
#define NUM_METRICS 2
const char *event_names[] = {
    "cuda_pw:::smsp__warps_launched.sum",
    "cuda_pw:::dram__bytes_write.sum",  //.pct_of_peak_burst_frame",
};

int main(int argc, char *argv[]) {
  int num_gpus = 0, i;
  CUcontext ctx_arr[NUM_THREADS];

  printf("%s Starting...\n\n", argv[0]);

  RUNTIME_API_CALL(cudaGetDeviceCount(&num_gpus));  // determine the number of CUDA capable GPUs

  if (num_gpus < 1) {
    printf("no CUDA capable devices were detected\n");
    return 1;
  }

  /////////////////////////////////////////////////////////////////
  // display CPU and GPU configuration
  //
  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of CUDA devices:\t%d\n", num_gpus);

  for (i = 0; i < num_gpus; i++) {
    cudaDeviceProp dprop;
    RUNTIME_API_CALL(cudaGetDeviceProperties(&dprop, i));
    printf("   %d: %s\n", i, dprop.name);
  }
  // Create a gpu context for every thread
  for (i=0; i < NUM_THREADS; i++) {
    DRIVER_API_CALL(cuCtxCreate(&(ctx_arr[i]), 0, i % num_gpus));  // "% num_gpus" allows more CPU threads than GPU devices
    DRIVER_API_CALL(cuCtxPopCurrent(&(ctx_arr[i])));
  }

  printf("---------------------------\n");
  int retval = PAPI_library_init( PAPI_VER_CURRENT );
  if( retval != PAPI_VER_CURRENT ) {
      fprintf( stderr, "Please recompile this test program. Installed PAPI has been updated.\n" );
      exit(-1);
  }
  PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num));
  omp_lock_t lock;
  omp_init_lock(&lock);

  omp_set_num_threads(NUM_THREADS);  // create as many CPU threads as there are CUDA devices
#pragma omp parallel
  {
    int EventSet = PAPI_NULL;
    long long values[NUM_METRICS];
    int j, res;
    PAPI_CALL(PAPI_create_eventset(&EventSet));
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();
    int gpu_id = cpu_thread_id % num_gpus;

    DRIVER_API_CALL(cuCtxPushCurrent(ctx_arr[cpu_thread_id]));
    printf("CPU thread %d (of %d) uses CUDA device %d with context %p @ eventset %d\n", cpu_thread_id, num_cpu_threads, gpu_id, ctx_arr[cpu_thread_id], EventSet);
    char tmpEventName[64];
    for (j=0; j<NUM_METRICS; j++) {
      snprintf(tmpEventName, 64, "%s:device=%d", event_names[j], gpu_id);
      fprintf(stderr, "Adding event name %s\n", tmpEventName);
      retval = PAPI_add_named_event( EventSet, tmpEventName );
      if (retval != PAPI_OK) {
        fprintf(stderr, "Error adding event %s\n", tmpEventName);
      }
    }
    PAPI_start(EventSet);

    VectorAddSubtract(50000*(cpu_thread_id+1));  // gpu work

    PAPI_stop(EventSet, values);
    printf("User measured values.\n");
    for (j=0; j<NUM_METRICS; j++) {
      snprintf(tmpEventName, 64, "%s:device=%d", event_names[j], gpu_id);
      printf("%s\t\t%lld\n", tmpEventName, values[j]);
    }
    DRIVER_API_CALL(cuCtxPopCurrent(&(ctx_arr[gpu_id])));
  }  // omp parallel region end

  printf("---------------------------\n");
  for (i=0; i < NUM_THREADS; i++) {
    DRIVER_API_CALL(cuCtxDestroy(ctx_arr[i]));
  }

  if (cudaSuccess != cudaGetLastError())
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  omp_destroy_lock(&lock);
  return 0;
}
