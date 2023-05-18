#ifndef __LCUDA_DISPATCH_H__
#define __LCUDA_DISPATCH_H__

#include "lcuda_common.h"

void cuptid_shutdown(void);
int cuptid_init(const char **disabled_reason);
int cuptid_thread_info_init(void **thread_info);
int cuptid_thread_info_free(void **thread_info);
int cuptid_control_create(event_list_t *event_names, int event_count, int *evt_ids, void **pcupti_ctl, void **pcu_ctx);
int cuptid_control_destroy(void **pcupti_ctl);
int cuptid_start(void *pcupti_ctl, void **pcu_ctx);
int cuptid_stop(void *pcupti_ctl, void **pcu_ctx);
int cuptid_control_read(void *pcupti_ctl, long long *values);
int cuptid_control_reset(void *pcupti_ctl);
int cuptid_enumerate_all_events(event_list_t *all_evt_names);
int cuptid_get_event_description(char *evt_name, char *descr);
#endif // __LCUDA_DISPATCH_H__
