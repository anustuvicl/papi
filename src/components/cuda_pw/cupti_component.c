#include "debug_comp.h"
#include "cuda_utils.h"
#include "cupti_component.h"

#include "cupti_profiler.h"
#include "cupti_events.h"

void cupti_shutdown(void)
{
    if (util_runtime_is_perfworks_api()) {
        cupti_profiler_shutdown();
    }
    util_unload_cuda_sym();
}

int cupti_init(const char **pdisabled_reason)
{
    int retval;
    retval = util_load_cuda_sym(pdisabled_reason);
    if (retval != PAPI_OK) {
        goto fn_exit;
    }
    if (util_runtime_is_perfworks_api())
        retval = cupti_profiler_init(pdisabled_reason);
    else if (util_runtime_is_events_api())
        retval = cupti_events_init(pdisabled_reason);
    else {
        *pdisabled_reason = "CUDA not found.";
        retval = PAPI_ECMP;
    }
fn_exit:
    return retval;
}

int cupti_thread_info_init(void **thread_info)
{
    return cucontext_array_init(thread_info);
}

int cupti_thread_info_free(void **thread_info)
{
    return cucontext_array_free(thread_info);
}

int cupti_control_create(event_list_t *event_names, int event_count, int *evt_ids, void **pcupti_ctl, void **pcu_ctx)
{
    if (util_runtime_is_perfworks_api()) {
        return cupti_profiler_control_create(event_names, event_count, evt_ids, pcupti_ctl, pcu_ctx);
    }
    return PAPI_ECMP;
}

int cupti_control_destroy(void **pcupti_ctl)
{
    if (util_runtime_is_perfworks_api()) {
        return cupti_profiler_control_destroy(pcupti_ctl);
    }
    return PAPI_ECMP;
}

int cupti_start(void *pcupti_ctl, void **pcu_ctx)
{
    if (util_runtime_is_perfworks_api()) {
        return cupti_profiler_start(pcupti_ctl, pcu_ctx);
    }
    else if (util_runtime_is_events_api()) {
        ;
    }
    return PAPI_ECMP;
}

int cupti_stop(void *pcupti_ctl, void **pcu_ctx)
{
    if (util_runtime_is_perfworks_api()) {
        return cupti_profiler_stop(pcupti_ctl, pcu_ctx);
    }
    else if (util_runtime_is_events_api()) {
        ;
    }
    return PAPI_ECMP;
}

int cupti_control_read(void *pcupti_ctl, long long *values)
{
    if (util_runtime_is_perfworks_api()) {
        return cupti_profiler_control_read(pcupti_ctl, values);
    }
    return PAPI_ECMP;
}

int cupti_control_reset(void *pcupti_ctl)
{
    if (util_runtime_is_perfworks_api())
        return cupti_profiler_control_reset(pcupti_ctl);
    return PAPI_ECMP;
}

int cupti_enumerate_all_events(event_list_t *all_evt_names)
{
    if (util_runtime_is_perfworks_api())
        return cupti_profiler_enumerate_all_metric_names(all_evt_names);
    return PAPI_ECMP;
}

int cupti_get_event_description(char *evt_name, char *descr)
{
    if (util_runtime_is_perfworks_api())
        return cupti_profiler_get_event_description(evt_name, descr);
    return PAPI_ECMP;
}
