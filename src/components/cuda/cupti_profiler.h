/**
 * @file    cupti_profiler.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_PROFILER_H__
#define __CUPTI_PROFILER_H__

#include "lcuda_common.h"

int cuptip_init(const char **pdisabled_reason);
int cuptip_control_create(event_list_t *all_event_names, int event_count, int *evt_ids, void **pctl, void **pcu_ctx);
int cuptip_control_destroy(void **pctl);
int cuptip_start(void **pctl, void **pcu_ctx);
int cuptip_stop(void **pctl, void **pcu_ctx);
int cuptip_control_read(void **pctl, long long *values);
int cuptip_control_reset(void **pctl);
int cuptip_enumerate_all_metric_names(event_list_t *all_evt_names);
int cuptip_get_event_description(const char *evt_name, char *description);
void cuptip_shutdown(void);
#endif  // __CUPTI_PROFILER_H__
