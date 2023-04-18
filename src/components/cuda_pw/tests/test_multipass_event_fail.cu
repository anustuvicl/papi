#include <stdio.h>
#include "papi.h"
#include "papi_test.h"

#define PASS 1
#define FAIL 0

int numEvents=3;
char const *EventName[] = {
    "cuda_pw:::smsp__warps_launched.sum:device=0",
    "cuda_pw:::dram__bytes_write.sum:device=0",
    "cuda_pw:::gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed:device=0"
};

int test_PAPI_add_named_event(int *EventSet) {
    int i, res;
    fprintf(stderr, "LOG: %s: Entering.\n", __func__);
    for (i=0; i<numEvents; i++) {
        res = PAPI_add_named_event(*EventSet, EventName[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Error %d: Failed to add event %s\n", res, EventName[i]);
            goto fail;
        }
    }
    if (res == PAPI_EMULPASS)
        return PASS;           // Test pass condition
fail:
    return FAIL;
}

int test_PAPI_add_event(int *EventSet) {
    int event, i, res;
    fprintf(stderr, "LOG: %s: Entering.\n", __func__);

    for (i=0; i<numEvents; i++) {
        res = PAPI_event_name_to_code(EventName[i], &event);
        if (res != PAPI_OK) {
            fprintf(stderr, "Error %d: Error in name to code.\n", res);
            goto fail;
        }
        res = PAPI_add_event(*EventSet, event);
        if (res != PAPI_OK) {
            fprintf(stderr, "Error %d: Failed to add event %s\n", res, EventName[i]);
            goto fail;
        }
    }
    if (res == PAPI_EMULPASS)
        return PASS;
fail:
    return FAIL;
}

int test_PAPI_add_events(int *EventSet) {
    int res, i;
    fprintf(stderr, "LOG: %s: Entering.\n", __func__);

    int events[numEvents];

    for (i=0; i<numEvents; i++) {
        res = PAPI_event_name_to_code(EventName[i], &events[i]);
        if (res != PAPI_OK) {
            fprintf(stderr, "Error %d: Error in name to code.\n", res);
            goto fail;
        }
    }
    res = PAPI_add_events(*EventSet, events, numEvents);
    if (res != PAPI_OK) {
        fprintf(stderr, "Error %d: Failed to add %d events\n", res, numEvents);
        goto fail;
    }
    if (res == 2)        // Returns index at which error occurred.
        return PASS;
fail:
    return FAIL;
}

int main() {
    int res, pass;
    int event_set;
    res = PAPI_library_init( PAPI_VER_CURRENT );
    if (res != PAPI_VER_CURRENT) test_fail(__FILE__, __LINE__, "PAPI_library_init() failed", 0);

    res = PAPI_get_component_index("cuda_pw");
    if (res < 0 ) test_fail(__FILE__, __LINE__, "CUDA_PERF component not configured", 0);

    event_set = PAPI_NULL;
    res = PAPI_create_eventset( &event_set );
    if (res != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    pass = test_PAPI_add_event(&event_set);
    res = PAPI_cleanup_eventset(event_set);
    res = PAPI_destroy_eventset(&event_set);

    event_set = PAPI_NULL;
    res = PAPI_create_eventset( &event_set );
    if (res != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    pass += test_PAPI_add_named_event(&event_set);
    res = PAPI_cleanup_eventset(event_set);
    res = PAPI_destroy_eventset(&event_set);

    event_set = PAPI_NULL;
    res = PAPI_create_eventset( &event_set );
    if (res != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    pass += test_PAPI_add_events(&event_set);
    res = PAPI_cleanup_eventset(event_set);
    res = PAPI_destroy_eventset(&event_set);

    if (pass != 3)
        test_fail(__FILE__, __LINE__, "CUDA framework multipass event test failed.", 0);
    else
        test_pass(__FILE__);

    PAPI_shutdown();
    return 0;
}