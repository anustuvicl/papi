#ifndef __CUPTI_COMPONENT_H__
#define __CUPTI_COMPONENT_H__

#include "common.h"

void cupti_shutdown(void);
int cupti_init(const char **disabled_reason);
int cupti_thread_info_init(void **thread_info);
int cupti_thread_info_free(void **thread_info);
int cupti_control_create(struct event_name_list_s * event_names, int event_count, int *evt_ids, void ** pcupti_ctl, void **pcu_ctx);
int cupti_control_destroy(void **pcupti_ctl);
int cupti_start(void *pcupti_ctl, void **pcu_ctx);
int cupti_stop(void *pcupti_ctl, void **pcu_ctx);
int cupti_control_read(void *pcupti_ctl, long long *values);
int cupti_enumerate_all_events(event_list_t *all_evt_names);
int cupti_get_event_description(char *evt_name, char *descr);
#endif // __CUPTI_COMPONENT_H__
