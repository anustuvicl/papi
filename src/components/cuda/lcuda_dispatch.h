/**
 * @file    lcuda_dispatch.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __LCUDA_DISPATCH_H__
#define __LCUDA_DISPATCH_H__

#include "lcuda_common.h"

void cuptid_shutdown(void);
int cuptid_init(const char **disabled_reason);
int cuptid_thread_info_create(void **pthread_info);
int cuptid_thread_info_destroy(void **pthread_info);
int cuptid_control_create(event_list_t *event_names, void *thread_info, void **pcupti_ctl);
int cuptid_control_destroy(void **pcupti_ctl);
int cuptid_start(void *cupti_ctl, void *thread_info);
int cuptid_stop(void *cupti_ctl, void *thread_info);
int cuptid_control_read(void *cupti_ctl, long long *values);
int cuptid_control_reset(void *cupti_ctl);
int cuptid_event_enum(event_list_t *all_evt_names);
int cuptid_get_event_description(char *evt_name, char *descr);
#endif // __LCUDA_DISPATCH_H__
