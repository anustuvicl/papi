#ifndef __CUPTI_EVENTS_H__
#define __CUPTI_EVENTS_H__

int cupti_events_init(const char **pdisabled_reason);
int cupti_events_control_create(struct event_name_list_s * all_event_names, int event_count, int *evt_ids, void ** pctl, void **pcu_ctx);
int cupti_events_control_destroy(void **pctl);
int cupti_events_start(void **pctl, void **pcu_ctx);
int cupti_events_stop(void **pctl, void **pcu_ctx);
int cupti_events_control_read(void **pctl, long long *values);
int cupti_events_control_reset(void **pctl);
int cupti_events_enumerate_all_metric_names(event_list_t *all_evt_names);
int cupti_events_get_event_description(const char *evt_name, char *description);
void cupti_events_shutdown(void);

#endif  // __CUPTI_EVENTS_H__
