#include <papi.h>
// #include "api_cuda.h"
// #include "cuda_common.h"

// Functions needed by CUPTI Events API
// ...

// CUPTI Events component API functions

int cupti_events_init(char **pdisabled_reason)
{
    *pdisabled_reason = "CUDA events API not implemented.";
    return PAPI_ECMP;
}
