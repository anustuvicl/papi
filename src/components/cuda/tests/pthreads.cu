#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "gpu_work.h"
#include <papi.h>
#include "papi_test.h"

#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}
int quiet;

#define PAPI_CALL(apiFuncCall)                                          \
do {                                                                           \
    int _status = apiFuncCall;                                         \
    if (_status != PAPI_OK) {                                              \
        fprintf(stderr, "error: function %s failed.", #apiFuncCall);  \
        test_fail(__FILE__, __LINE__, "", _status);  \
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

#define MAX_THREADS (32)

int numGPUs;
int g_event_count;
char **g_evt_names;

pthread_t tidarr[MAX_THREADS];
CUcontext cuCtx[MAX_THREADS];
pthread_mutex_t lock;

void *thread_gpu(void * idx)
{
    int tid = *((int*) idx);
    int gpuid = tid % numGPUs;
    unsigned long gettid = (unsigned long) pthread_self();
    int papi_errno, i;

    int EventSet = PAPI_NULL;
    long long values[MAX_THREADS];
    PAPI_CALL(PAPI_create_eventset(&EventSet));

    DRIVER_API_CALL(cuCtxSetCurrent(cuCtx[tid]));
    PRINT(quiet, "This is idx %d thread %lu - using GPU %d context %p!\n",
            tid, gettid, gpuid, cuCtx[tid]);

    char tmpEventName[64];
    for (i = 0; i < g_event_count; i++) {
        snprintf(tmpEventName, 64, "%s:device=%d", g_evt_names[i], gpuid);
        papi_errno = PAPI_add_named_event(EventSet, tmpEventName);
        if (papi_errno != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", tmpEventName);
        }
    }

    PAPI_CALL(PAPI_start(EventSet));

    VectorAddSubtract(50000*(tid+1), quiet);  // gpu work

    PAPI_CALL(PAPI_stop(EventSet, values));

    PRINT(quiet, "User measured values in thread id %d.\n", tid);
    for (i = 0; i < g_event_count; i++) {
        snprintf(tmpEventName, 64, "%s:device=%d", g_evt_names[i], gpuid);
        PRINT(quiet, "%s\t\t%lld\n", tmpEventName, values[i]);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    quiet = 0;
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    g_event_count = argc - 1;
    /* if no events passed at command line, just report test skipped. */
    if (g_event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    g_evt_names = argv + 1;

    int rc, i;
    int tid[MAX_THREADS];

    RUNTIME_API_CALL(cudaGetDeviceCount(&numGPUs));
    PRINT(quiet, "No. of GPUs = %d\n", numGPUs);
    if (numGPUs < 1) {
        fprintf(stderr, "No GPUs found on system.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }
    if (numGPUs > MAX_THREADS)
        numGPUs = MAX_THREADS;
    PRINT(quiet, "No. of threads to launch = %d\n", numGPUs);

    int papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed.", 0);
    }
    // Point PAPI to function that gets the thread id
    PAPI_CALL(PAPI_thread_init((unsigned long (*)(void)) pthread_self));

    // Launch the threads
    for(i = 0; i < numGPUs; i++)
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
        PRINT(quiet, "\n Main thread %lu. Created new thread (%lu) in iteration %d ...\n",
                (unsigned long)pthread_self(), (unsigned long)tidarr[i], i);
    }

    // Join all threads when complete
    for (i = 0; i < numGPUs; i++) {
        pthread_join(tidarr[i], NULL);
        PRINT(quiet, "IDX: %d: TID: %lu: Done! Joined main thread.\n", i, (unsigned long)tidarr[i]);
    }

    // Destroy all CUDA contexts for all threads/GPUs
    for (i = 0; i < numGPUs; i++) {
        DRIVER_API_CALL(cuCtxDestroy(cuCtx[i]));
    }
    PRINT(quiet, "Main thread exit!\n");
    test_pass(__FILE__);
    return 0;
}
