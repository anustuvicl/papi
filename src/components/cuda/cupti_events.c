#include <papi.h>
#include "lcuda_common.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"
// Functions needed by CUPTI Events API
// ...

// CUPTI Events component API functions

int cuptie_init(char **pdisabled_reason)
{
    *pdisabled_reason = "CUDA events API not implemented.";
    return PAPI_ENOIMPL;
}

int cuptie_control_create(event_list_t *all_event_names, int event_count, int *evt_ids, void **pctl, void **pcu_ctx)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_destroy(void **pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_start(void **pctl, void **pcu_ctx)
{
    return PAPI_ENOIMPL;
}

int cuptie_stop(void **pctl, void **pcu_ctx)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_read(void **pctl, long long *values)
{
    return PAPI_ENOIMPL;
}

int cuptie_control_reset(void **pctl)
{
    return PAPI_ENOIMPL;
}

int cuptie_enumerate_all_metric_names(event_list_t *all_evt_names)
{
    return PAPI_ENOIMPL;
}

int cuptie_get_event_description(const char *evt_name, char *description)
{
    return PAPI_ENOIMPL;
}

void cuptie_shutdown(void)
{
    return;
}
