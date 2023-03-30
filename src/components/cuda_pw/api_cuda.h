/*
    Include appropriate CUDA APIs. Set flags to check which API.
*/
#ifndef __SELECT_API_H__
#define __SELECT_API_H__

    #include <cuda.h>
    #include <cupti.h>

    #define API_PERFWORKS 0
    #define API_EVENTS 0
    #define API_ERROR 1

    #if CUPTI_API_VERSION >= 13 && CUDA_VERSION >= 11000
        #undef API_ERROR
        #undef API_PERFWORKS

        #define API_PERFWORKS 1
        #define API_ERROR 0

        // Include these files if CUDA profiler API
        #include <cupti_target.h>
        #include <cupti_profiler_target.h>
        #include <nvperf_host.h>
        #include <nvperf_cuda_host.h>
        #include <nvperf_target.h>
    #endif

    #if CUPTI_API_VERSION < 17 && CUDA_VERSION <= 11700
        #undef API_ERROR
        #undef API_EVENTS

        #define API_EVENTS 0  // Set this to 1 to activate Events API
        #define API_ERROR 0
    #endif

#endif  // __SELECT_API_H__