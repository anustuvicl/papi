#include <stdio.h>
#include <papi.h>
#include "gpu_work.h"

#define COMP_NAME "cuda_pw"

#define NUM_METRICS 5
const char *test_metrics[] = {
    COMP_NAME ":::smsp__warps_launched.sum:device=0",
    COMP_NAME ":::dram__bytes_read.min:device=0",
    COMP_NAME ":::dram__bytes_read.max:device=0",
    COMP_NAME ":::dram__bytes_read.sum:device=0",
    COMP_NAME ":::dram__bytes_write.sum:device=0"
};

void multi_read(long long *values)
{
    int EventSet = PAPI_NULL;
    int res, i, j;
    CUcontext ctx;
    res = PAPI_create_eventset(&EventSet);
    res = cuCtxCreate(&ctx, 0, 0);
    for (i=0; i<NUM_METRICS; i++) {
        res = PAPI_add_named_event(EventSet, test_metrics[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", test_metrics[i]);
        }
    }

    res = PAPI_start(EventSet);

    for (i=0; i<10; i++) {
        VectorAddSubtract(50000*(i+1));
        res = PAPI_read(EventSet, values);
        printf("Measured values iter %d\n", i);
        for (j=0; j<NUM_METRICS; j++) {
            printf("%s\t\t%lld\n", test_metrics[j], values[j]);
        }
    }
    res = PAPI_stop(EventSet, values);
    res = PAPI_cleanup_eventset(EventSet);
    res = cuCtxDestroy(ctx);
}

void single_read(long long *values)
{
    int EventSet = PAPI_NULL;
    int res, i, j;
    CUcontext ctx;
    res = PAPI_create_eventset(&EventSet);
    res = cuCtxCreate(&ctx, 0, 0);
    for (i=0; i<NUM_METRICS; i++) {
        res = PAPI_add_named_event(EventSet, test_metrics[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", test_metrics[i]);
        }
    }

    res = PAPI_start(EventSet);

    for (i=0; i<10; i++) {
        VectorAddSubtract(50000*(i+1));
    }
    res = PAPI_stop(EventSet, values);
    printf("Measured values from single read\n");
    for (j=0; j<NUM_METRICS; j++) {
        printf("%s\t\t%lld\n", test_metrics[j], values[j]);
    }
    res = PAPI_cleanup_eventset(EventSet);
    res = cuCtxDestroy(ctx);
}

int main()
{
    int res;
    res = PAPI_library_init(PAPI_VER_CURRENT);

    res = PAPI_get_component_index(COMP_NAME);
    if (res < 0) {
        fprintf(stderr, "PAPI not configured with '" COMP_NAME "' component!");
        exit(-1);
    }
    long long values_multi_read[NUM_METRICS], values_single_read[NUM_METRICS];

    multi_read(values_multi_read);
    single_read(values_single_read);

    int i;
    printf("Final measured values\nEvent_name\t\t\t\t\t\tMulti_read\tsingle_read\n");
    for (i=0; i<NUM_METRICS; i++) {
        printf("%s\t\t\t%lld\t\t%lld\n", test_metrics[i], values_multi_read[i], values_single_read[i]);
    }
    PAPI_shutdown();
    return 0;
}