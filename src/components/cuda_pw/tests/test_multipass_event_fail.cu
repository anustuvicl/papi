#include <stdio.h>
#include "papi.h"
#include "papi_test.h"

int numEvents=3;
char const *EventName[] = {
    "cuda_pw:::smsp__warps_launched.sum:device=0",
    "cuda_pw:::dram__bytes_write.sum:device=0",
    "cuda_pw:::gpu__compute_memory_access_throughput_internal_activity.max.pct_of_peak_sustained_elapsed:device=0"
};

int test_PAPI_add_named_event(int *EventSet) {
    int i, retval;
    fprintf(stderr, "LOG: %s: Entering.\n", __func__);
    for (i=0; i<numEvents; i++) {
        retval = PAPI_add_named_event(*EventSet, EventName[i]);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error %d: Failed to add event %s\n", retval, EventName[i]);
        }
    }
    if (retval == PAPI_EMULPASS)
        return 1;           // Test pass condition
    return 0;
}

int test_PAPI_add_event(int *EventSet) {
    int event, i, retval;
    fprintf(stderr, "LOG: %s: Entering.\n", __func__);

    for (i=0; i<numEvents; i++) {
        retval = PAPI_event_name_to_code(EventName[i], &event);
        retval = PAPI_add_event(*EventSet, event);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error %d: Failed to add event %s\n", retval, EventName[i]);
        }
    }
    if (retval == PAPI_EMULPASS)
        return 1;
    return 0;
}

int test_PAPI_add_events(int *EventSet) {
    int retval, i;
    fprintf(stderr, "LOG: %s: Entering.\n", __func__);

    int events[numEvents];

    for (i=0; i<numEvents; i++) {
        retval = PAPI_event_name_to_code(EventName[i], &events[i]);
    }
    retval = PAPI_add_events(*EventSet, events, numEvents);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Error %d: Failed to add %d events\n", retval, numEvents);
    }
    if (retval == 2)        // Returns index at which error occurred.
        return 1;
    return 0;
}

int main() {
    int retval, pass;
    int event_set;
    retval = PAPI_library_init( PAPI_VER_CURRENT );
    if (retval != PAPI_VER_CURRENT) test_fail(__FILE__, __LINE__, "PAPI_library_init() failed", 0);

    retval = PAPI_get_component_index("cuda_pw");
    if (retval < 0 ) test_fail(__FILE__, __LINE__, "CUDA_PERF component not configured", 0);

    event_set = PAPI_NULL;
    retval = PAPI_create_eventset( &event_set );
    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    pass = test_PAPI_add_event(&event_set);
    retval = PAPI_cleanup_eventset(event_set);
    retval = PAPI_destroy_eventset(&event_set);

    event_set = PAPI_NULL;
    retval = PAPI_create_eventset( &event_set );
    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    pass += test_PAPI_add_named_event(&event_set);
    retval = PAPI_cleanup_eventset(event_set);
    retval = PAPI_destroy_eventset(&event_set);

    event_set = PAPI_NULL;
    retval = PAPI_create_eventset( &event_set );
    if (retval != PAPI_OK) test_fail(__FILE__, __LINE__, "PAPI_create_eventset() failed!", 0);

    pass += test_PAPI_add_events(&event_set);
    retval = PAPI_cleanup_eventset(event_set);
    retval = PAPI_destroy_eventset(&event_set);

    if (pass != 3)
        test_fail(__FILE__, __LINE__, "CUDA framework multipass event test failed.", 0);
    else
        test_pass(__FILE__);

    PAPI_shutdown();
    return 0;
}