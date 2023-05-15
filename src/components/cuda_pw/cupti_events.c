#include <papi.h>
// #include "api_cuda.h"
#include "common.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
// Functions needed by CUPTI Events API
// ...

// CUPTI Events component API functions

int cupti_events_init(char **pdisabled_reason)
{
    *pdisabled_reason = "CUDA events API not implemented.";
    return PAPI_ENOIMPL;
}

int cupti_events_control_create(event_list_t * all_event_names, int event_count, int *evt_ids, void ** pctl, void **pcu_ctx)
{
    return PAPI_ENOIMPL;
}

int cupti_events_control_destroy(void **pctl)
{
    return PAPI_ENOIMPL;
}

int cupti_events_start(void **pctl, void **pcu_ctx)
{
    return PAPI_ENOIMPL;
}

int cupti_events_stop(void **pctl, void **pcu_ctx)
{
    return PAPI_ENOIMPL;
}

int cupti_events_control_read(void **pctl, long long *values)
{
    return PAPI_ENOIMPL;
}

int cupti_events_control_reset(void **pctl)
{
    return PAPI_ENOIMPL;
}

int cupti_events_enumerate_all_metric_names(event_list_t *all_evt_names)
{
    return PAPI_ENOIMPL;
}

int cupti_events_get_event_description(const char *evt_name, char *description)
{
    return PAPI_ENOIMPL;
}

void cupti_events_shutdown(void)
{
    return;
}
