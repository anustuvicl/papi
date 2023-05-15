#ifndef __CUDA_PROFILER_H__
#define __CUDA_PROFILER_H__

#include "common.h"

int cupti_profiler_init(const char ** pdisabled_reason);
int cupti_profiler_control_create(event_list_t * all_event_names, int event_count, int *evt_ids, void ** pctl, void **pcu_ctx);
int cupti_profiler_control_destroy(void **pctl);
int cupti_profiler_start(void **pctl, void **pcu_ctx);
int cupti_profiler_stop(void **pctl, void **pcu_ctx);
int cupti_profiler_control_read(void **pctl, long long *values);
int cupti_profiler_control_reset(void **pctl);
int cupti_profiler_enumerate_all_metric_names(event_list_t *all_evt_names);
int cupti_profiler_get_event_description(const char *evt_name, char *description);
void cupti_profiler_shutdown(void);
#endif  // __CUDA_PROFILER_H__
