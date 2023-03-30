#include <papi.h>
#include <stdio.h>

#define LOG(...) \
do { \
fprintf(stderr, "LOG MAIN: ln %d: ", __LINE__); \
fprintf(stderr, __VA_ARGS__); \
} while (0);

#define NUM_EVENTS 7
char const *EventNames[] = {
    "cuda_pw:::dram__bytes_read.sum:device=0",
    "cuda_pw:::fe__cycles_elapsed.sum:device=0",
    "cuda_pw:::sm__cycles_active.sum:device=0",
    "cuda_pw:::sm__threads_launched.su:device=0",
    "cuda_pw:::smsp__cycles_active.sum:device=0",
    "cuda_pw:::smsp__inst_executed.sum:device=0",
    "cuda_pw:::sys__cycles_elapsed.sum:device=0",
};

int main()
{
    int retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf( stderr, "Please recompile this test program. Installed PAPI has been updated.\n" );
        exit(-1);
    }

    int cid = PAPI_get_component_index("cuda_pw");
    LOG("cuda_pw component index = %d\n", cid);

    int i, evtcode;
    char code2name[64];
    for (i=0; i<NUM_EVENTS; i++) {
        retval = PAPI_event_name_to_code(EventNames[i], &evtcode);
        printf("Evt name = %s, evtcode = %d\n", EventNames[i], evtcode);
        retval = PAPI_event_code_to_name(evtcode, code2name);
        printf("Event code to name = %s\n", code2name);

    }

    PAPI_shutdown();
    return 0;
}