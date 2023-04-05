#include <papi.h>
#include <papi_internal.h>
#include <papi_vector.h>

#include "debug_comp.h"
#include "cuda_api_config.h"
#include "common.h"
#include "cuda_utils.h"
#include "cupti_component.h"

#if defined API_PERFWORKS
#include "cupti_profiler.h"
#elif defined API_EVENTS
#include "cupti_events.h"
#endif

extern papi_vector_t _cuda_pw_vector;

int cupti_shutdown(void)
{
    return PAPI_OK;
}

int cupti_init(const char **pdisabled_reason)
{
    int retval;
#if defined API_PERFWORKS
    retval = cupti_profiler_init(pdisabled_reason);
#elif defined API_EVENTS
    retval = cupti_events_init(pdisabled_reason);
#else
    *pdisabled_reason = "CUDA not found.";
    retval = PAPI_ECMP;
#endif
    if (retval != PAPI_OK)
    {
        strcpy(_cuda_pw_vector.cmp_info.disabled_reason, *pdisabled_reason);
        goto fn_fail;
    }
    _cuda_pw_vector.cmp_info.disabled = PAPI_OK;
    return PAPI_OK;
fn_fail:
    return PAPI_ECMP;
}

int cupti_init_cuctx_arr(void ** pcuda_context)
{
    return init_CUcontext_array(pcuda_context);
}

int cupti_control_create(event_list_t * event_names, int event_count, int *evt_ids, void ** pcupti_ctl, void **pcu_ctx)
{
#if defined API_PERFWORKS
    int res = cupti_profiler_control_create(event_names, event_count, evt_ids, pcupti_ctl, pcu_ctx);
    return res;
#endif
}

int cupti_control_destroy(void **pcupti_ctl)
{
#if defined API_PERFWORKS
    int res = cupti_profiler_control_destroy(pcupti_ctl);
    return res;
#endif
}

int cupti_start(void *pcupti_ctl, void **pcu_ctx)
{
#if defined API_PERFWORKS
    return cupti_profiler_start(pcupti_ctl, pcu_ctx);
#elif defined API_EVENTS
    //
#endif
    (void)pcupti_ctl;
    return PAPI_OK;
}

int cupti_stop(void *pcupti_ctl, void **pcu_ctx){  // NOTE: void **pcupti_ctl ??
#if defined API_PERFWORKS
    return cupti_profiler_stop(pcupti_ctl, pcu_ctx);
#elif defined API_EVENTS
    //
#endif
    (void) pcupti_ctl;
    return PAPI_OK;
}

int cupti_control_read(void *pcupti_ctl, long long *values)
{
#if defined API_PERFWORKS
    return cupti_profiler_control_read(pcupti_ctl, values);
#endif
}

int cupti_control_reset(void *pcupti_ctl);

int cupti_enumerate_all_events(event_list_t *all_evt_names)
{
#if defined API_PERFWORKS
    return cupti_profiler_enumerate_all_metric_names(all_evt_names);
#endif
}
