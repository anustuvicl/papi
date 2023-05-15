#include "cuda_api_config.h"
#include "debug_comp.h"
#include "cuda_utils.h"
#include "cupti_component.h"

#if defined(API_PERFWORKS)
#   include "cupti_profiler.h"
#endif
#if defined(API_EVENTS)
#   include "cupti_events.h"
#endif

void cupti_shutdown(void)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        cupti_profiler_shutdown();
#endif
    }
    else if(util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        cupti_events_shutdown();
#endif
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

    if (util_gpu_collection_kind() == GPU_COLLECTION_MIXED) {
        *pdisabled_reason = "No support for systems with mixed compute capabilities, such as CC < 7.0 and CC > 7.0 GPUS.";
        retval = PAPI_ECMP;
        goto fn_exit;
    }

    if (util_runtime_is_perfworks_api()) {

#if defined(API_PERFWORKS)
        retval = cupti_profiler_init(pdisabled_reason);
#else
        *pdisabled_reason = "PAPI not built with NVIDIA profiler API support.";
        retval = PAPI_ECMP;
        goto fn_exit;
#endif

    } else if (util_runtime_is_events_api()) {

#if defined(API_EVENTS)
        retval = cupti_events_init(pdisabled_reason);
#else
        *pdisabled_reason = "Unknown events API problem.";
        retval = PAPI_ECMP;
#endif

    } else {
        *pdisabled_reason = "CUDA configuration not supported.";
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
#if defined(API_PERFWORKS)
        return cupti_profiler_control_create(event_names, event_count, evt_ids, pcupti_ctl, pcu_ctx);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined (API_EVENTS)
        return cupti_events_control_create(event_names, event_count, evt_ids, pcupti_ctl, pcu_ctx);
#endif
    }
    return PAPI_ECMP;
}

int cupti_control_destroy(void **pcupti_ctl)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cupti_profiler_control_destroy(pcupti_ctl);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_control_destroy(pcupti_ctl);
#endif
    }
    return PAPI_ECMP;
}

int cupti_start(void *pcupti_ctl, void **pcu_ctx)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cupti_profiler_start(pcupti_ctl, pcu_ctx);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_start(pcupti_ctl, pcu_ctx);
#endif
    }
    return PAPI_ECMP;
}

int cupti_stop(void *pcupti_ctl, void **pcu_ctx)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cupti_profiler_stop(pcupti_ctl, pcu_ctx);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_stop(pcupti_ctl, pcu_ctx);
#endif
    }
    return PAPI_ECMP;
}

int cupti_control_read(void *pcupti_ctl, long long *values)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cupti_profiler_control_read(pcupti_ctl, values);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_control_read(pcupti_ctl, values);
#endif
    }
    return PAPI_ECMP;
}

int cupti_control_reset(void *pcupti_ctl)
{
    if (util_runtime_is_perfworks_api())
#if defined(API_PERFWORKS)
        return cupti_profiler_control_reset(pcupti_ctl);
#endif
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_control_reset(pcupti_ctl);
#endif
    }
    return PAPI_ECMP;
}

int cupti_enumerate_all_events(event_list_t *all_evt_names)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cupti_profiler_enumerate_all_metric_names(all_evt_names);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_enumerate_all_metric_names(all_evt_names);
#endif
    }
    return PAPI_ECMP;
}

int cupti_get_event_description(char *evt_name, char *descr)
{
    if (util_runtime_is_perfworks_api()) {
#if defined(API_PERFWORKS)
        return cupti_profiler_get_event_description(evt_name, descr);
#endif
    }
    else if (util_runtime_is_events_api()) {
#if defined(API_EVENTS)
        return cupti_events_get_event_description(evt_name, descr);
#endif
    }
    return PAPI_ECMP;
}
