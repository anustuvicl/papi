#ifndef __CUPTI_EVENTS_H__
#define __CUPTI_EVENTS_H__

#include "lcuda_common.h"

int cuptie_init(const char **pdisabled_reason);
int cuptie_control_create(event_list_t *all_event_names, int event_count, int *evt_ids, void **pctl, void **pcu_ctx);
int cuptie_control_destroy(void **pctl);
int cuptie_start(void **pctl, void **pcu_ctx);
int cuptie_stop(void **pctl, void **pcu_ctx);
int cuptie_control_read(void **pctl, long long *values);
int cuptie_control_reset(void **pctl);
int cuptie_enumerate_all_metric_names(event_list_t *all_evt_names);
int cuptie_get_event_description(const char *evt_name, char *description);
void cuptie_shutdown(void);

#endif  // __CUPTI_EVENTS_H__
