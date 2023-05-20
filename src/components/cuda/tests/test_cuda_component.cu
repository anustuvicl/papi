#include <papi.h>
#include <stdio.h>
#include <cuda.h>

#define LOG(...) \
do { \
fprintf(stderr, "LOG MAIN: ln %d: ", __LINE__); \
fprintf(stderr, __VA_ARGS__); \
} while (0);

#define NUM_GPUS 8
#define NUM_EVENTS 7
char const *EventNames[] = {
    "cuda:::dram__bytes_read.sum",
    "cuda:::fe__cycles_elapsed.sum",
    "cuda:::sm__cycles_active.sum",
    "cuda:::sm__threads_launched.sum",
    "cuda:::smsp__cycles_active.sum",
    "cuda:::smsp__inst_executed.sum",
    "cuda:::sys__cycles_elapsed.sum",
};

int main()
{
    int papi_errno;

    cuInit(0);

    papi_errno = PAPI_library_init( PAPI_VER_CURRENT );
    if( papi_errno != PAPI_VER_CURRENT ) {
        fprintf( stderr, "Please recompile this test program. Installed PAPI has been updated.\n" );
        exit(-1);
    }

    int cid = PAPI_get_component_index("cuda");
    LOG("cuda component index = %d\n", cid);

    int eventset = PAPI_NULL;
    papi_errno = PAPI_create_eventset(&eventset);

    /* Framework does not call update_control_state, the first time
    an event is added. It only calls init_control_state. So the
    first event name gets missed. Only way to fix that is to call
    PAPI_assign_eventset_component first. This might be a potential
    bug. */
    papi_errno = PAPI_assign_eventset_component(eventset, cid);

    CUcontext cuctx;
    char tmpName[128];
    int n, i;
    for (n=0; n<NUM_GPUS; n++) {
        papi_errno = cuCtxCreate(&cuctx, 0, n);
        LOG("Context created %p for device %d\n", cuctx, n);
        for (i=0; i<NUM_EVENTS; i++) {
            sprintf(tmpName, "%s:device=%d", EventNames[i], n);
            LOG("Adding event %s\n", tmpName);
            papi_errno = PAPI_add_named_event(eventset, tmpName);
            if (papi_errno != PAPI_OK) {
                fprintf(stderr, "Error %d: Failed to add event %s\n", tmpName);
            }
        }
    }

    papi_errno = PAPI_start(eventset);

    // Kernel launches here

    long long values[NUM_EVENTS*NUM_GPUS];
    papi_errno = PAPI_stop(eventset, values);

    PAPI_shutdown();
    return 0;
}