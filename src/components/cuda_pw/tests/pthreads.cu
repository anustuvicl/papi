#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gpu_work.h"
#include <papi.h>

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
const char *test_metrics[] = {
    "cuda_pw:::smsp__warps_launched.sum",
    "cuda_pw:::dram__bytes_write.sum",
};

int numGPUs;

pthread_t tidarr[NUM_THREADS];
CUcontext cuCtx[NUM_THREADS];
pthread_mutex_t lock;

void * thread_gpu(void * idx)
{
    int tid = *((int*) idx);
    int gpuid = tid % numGPUs;
    unsigned long gettid = (unsigned long) pthread_self();
    int retval, i;

    int EventSet = PAPI_NULL;
    long long values[NUM_METRICS];
    PAPI_CALL(PAPI_create_eventset(&EventSet));

    DRIVER_API_CALL(cuCtxSetCurrent(cuCtx[tid]));
    fprintf(stderr, "This is idx %d thread %lu - using GPU %d context %p!\n",
            tid, gettid, gpuid, cuCtx[tid]);

    char tmpEventName[64];
    for (i=0; i<NUM_METRICS; i++) {
        snprintf(tmpEventName, 64, "%s:device=%d", test_metrics[i], gpuid);
        retval = PAPI_add_named_event(EventSet, tmpEventName);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", tmpEventName);
        }
    }

    PAPI_CALL(PAPI_start(EventSet));
    
    VectorAddSubtract(50000*(tid+1));  // gpu work
    
    PAPI_CALL(PAPI_stop(EventSet, values));

    printf("User measured values in thread id %d.\n", tid);
    for (i=0; i<NUM_METRICS; i++) {
        snprintf(tmpEventName, 64, "%s:device=%d", test_metrics[i], gpuid);
        printf("%s\t\t%lld\n", tmpEventName, values[i]);
    }
    return NULL;
}

int main()
{
    int rc, i;
    int tid[NUM_THREADS];
    RUNTIME_API_CALL(cudaGetDeviceCount(&numGPUs));
    printf("No. of GPUs = %d\n", numGPUs);
    printf("No. of threads to launch = %d\n", NUM_THREADS);

    int retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf( stderr, "PAPI_library_init failed\n" );
        exit(-1);
    }
    // Point PAPI to function that gets the thread id
    PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) pthread_self));

    // Launch the threads
    for(i = 0; i < NUM_THREADS; i++)
    {
        tid[i] = i;
        DRIVER_API_CALL(cuCtxCreate(&(cuCtx[i]), 0, i % numGPUs));
        DRIVER_API_CALL(cuCtxPopCurrent(&(cuCtx[i])));

        rc = pthread_create(&tidarr[i], NULL, thread_gpu, &(tid[i]));
        if(rc)
        {
            fprintf(stderr, "\n ERROR: return code from pthread_create is %d \n", rc);
            exit(1);
        }
        fprintf(stderr, "\n Main thread %lu. Created new thread (%lu) in iteration %d ...\n",
                (unsigned long)pthread_self(), (unsigned long)tidarr[i], i);
    }

    // Join all threads when complete
    for (i=0; i<NUM_THREADS; i++) {
        pthread_join(tidarr[i], NULL);
        fprintf(stderr, "IDX: %d: TID: %lu: Done! Joined main thread.\n", i, (unsigned long)tidarr[i]);
    }

    // Destroy all CUDA contexts for all threads/GPUs
    for (i=0; i<NUM_THREADS; i++) {
        DRIVER_API_CALL(cuCtxDestroy(cuCtx[i]));
    }
    printf("Main thread exit!\n");
    return 0;
}
