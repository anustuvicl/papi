#include <stdio.h>
#include <papi.h>
#include "gpu_work.h"

#define COMP_NAME "cuda_pw"

#define NUM_METRICS 1
const char *test_metrics[] = {
    COMP_NAME ":::smsp__warps_launched.sum:device=0",
};

void multi_reset(long long *values)
{
    int EventSet = PAPI_NULL;
    int res, i, j;
    CUcontext ctx;
    res = PAPI_create_eventset(&EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "Failed to create eventset.\n");
    }

    res = cuCtxCreate(&ctx, 0, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to create cuda context.\n");
    }

    for (i=0; i<NUM_METRICS; i++) {
        res = PAPI_add_named_event(EventSet, test_metrics[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", test_metrics[i]);
        }
    }

    res = PAPI_start(EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_start error.\n");
    }
    for (i=0; i<10; i++) {
        VectorAddSubtract(100000);
        res = PAPI_read(EventSet, values);
        if (res != PAPI_OK) {
            fprintf(stderr, "PAPI_read error.\n");
        }
        printf("Measured values iter %d\n", i);
        for (j=0; j<NUM_METRICS; j++) {
            printf("%s\t\t%lld\n", test_metrics[j], values[j]);
        }
        res = PAPI_reset(EventSet);
    }
    res = PAPI_stop(EventSet, values);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_stop error.\n");
    }
    res = PAPI_cleanup_eventset(EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_cleanup_eventset error.\n");
    }
    res = cuCtxDestroy(ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cude error: failed to destroy context.\n");
    }
}

void multi_read(long long *values)
{
    int EventSet = PAPI_NULL;
    int res, i, j;
    CUcontext ctx;
    res = PAPI_create_eventset(&EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "Failed to create eventset.\n");
    }

    res = cuCtxCreate(&ctx, 0, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to create cuda context.\n");
    }

    for (i=0; i<NUM_METRICS; i++) {
        res = PAPI_add_named_event(EventSet, test_metrics[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", test_metrics[i]);
        }
    }

    res = PAPI_start(EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_start error.\n");
    }
    for (i=0; i<10; i++) {
        VectorAddSubtract(100000);
        res = PAPI_read(EventSet, values);
        if (res != PAPI_OK) {
            fprintf(stderr, "PAPI_read error.\n");
        }
        printf("Measured values iter %d\n", i);
        for (j=0; j<NUM_METRICS; j++) {
            printf("%s\t\t%lld\n", test_metrics[j], values[j]);
        }
    }
    res = PAPI_stop(EventSet, values);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_stop error.\n");
    }
    res = PAPI_cleanup_eventset(EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_cleanup_eventset error.\n");
    }
    res = cuCtxDestroy(ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cude error: failed to destroy context.\n");
    }
}

void single_read(long long *values)
{
    int EventSet = PAPI_NULL;
    int res, i, j;
    CUcontext ctx;
    res = PAPI_create_eventset(&EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_create_eventset error.\n");
    }
    res = cuCtxCreate(&ctx, 0, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to create cuda context.\n");
    }
    for (i=0; i<NUM_METRICS; i++) {
        res = PAPI_add_named_event(EventSet, test_metrics[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", test_metrics[i]);
        }
    }

    res = PAPI_start(EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_start error.\n");
    }
    for (i=0; i<10; i++) {
        VectorAddSubtract(100000);
    }
    res = PAPI_stop(EventSet, values);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_stop error.\n");
    }
    printf("Measured values from single read\n");
    for (j=0; j<NUM_METRICS; j++) {
        printf("%s\t\t%lld\n", test_metrics[j], values[j]);
    }
    res = PAPI_cleanup_eventset(EventSet);
    if (res != PAPI_OK) {
        fprintf(stderr, "PAPI_cleanup_eventset error.\n");
    }
    res = cuCtxDestroy(ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuda error: failed to destroy cuda context.\n");
    }
}

int main()
{
    int res;
    res = cuInit(0);

    res = PAPI_library_init(PAPI_VER_CURRENT);
    if (res != PAPI_VER_CURRENT) {
        fprintf(stderr, "Failed to initialize PAPI.\n");
    }
    res = PAPI_get_component_index(COMP_NAME);
    if (res < 0) {
        fprintf(stderr, "PAPI not configured with '" COMP_NAME "' component!");
        exit(-1);
    }
    long long values_multi_reset[NUM_METRICS];
    long long values_multi_read[NUM_METRICS];
    long long values_single_read[NUM_METRICS];

    printf("Running multi_reset.\n");
    multi_reset(values_multi_reset);
    printf("\nRunning multi_read.\n");
    multi_read(values_multi_read);
    printf("\nRunning single_read.\n");
    single_read(values_single_read);

    int i;
    printf("Final measured values\nEvent_name\t\t\t\t\t\tMulti_read\tsingle_read\n");
    for (i=0; i<NUM_METRICS; i++) {
        printf("%s\t\t\t%lld\t\t%lld\n", test_metrics[i], values_multi_read[i], values_single_read[i]);
    }
    PAPI_shutdown();
    return 0;
}
