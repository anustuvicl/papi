/**
 * @file    cupti_profiler.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUPTI_PROFILER_H__
#define __CUPTI_PROFILER_H__

#include "lcuda_common.h"

int cuptip_init(const char **pdisabled_reason);
int cuptip_control_create(ntv_event_table_t *event_names, void *thr_info, void **pctl);
int cuptip_control_destroy(void **pctl);
int cuptip_control_start(void *ctl, void *thr_info);
int cuptip_control_stop(void *ctl, void *thr_info);
int cuptip_control_read(void *ctl, long long *values);
int cuptip_control_reset(void *ctl);
int cuptip_event_enum(ntv_event_table_t *all_evt_names);
int cuptip_get_event_description(const char *evt_name, char *description);
void cuptip_shutdown(void);
#endif  // __CUPTI_PROFILER_H__
