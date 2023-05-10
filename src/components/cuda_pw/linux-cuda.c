#include <papi.h>
#include <papi_internal.h>
#include <papi_vector.h>

#include <string.h>

#include "debug_comp.h"
#include "common.h"

#include "cupti_component.h"

papi_vector_t _cuda_pw_vector;

static int cuda_init_component(int cidx);
static int cuda_shutdown_component(void);
static int cuda_init_thread(hwd_context_t *ctx);
static int cuda_shutdown_thread(hwd_context_t *ctx);

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier);
static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code);
static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len);

static int cuda_init_control_state(hwd_control_state_t *ctl);
static int cuda_set_domain(hwd_control_state_t * ctrl, int domain);
static int cuda_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, hwd_context_t *ctx);

static int cuda_cleanup_eventset(hwd_control_state_t *ctl);
static int cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int cuda_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_init_private(void);

#define PAPI_CUDA_MAX_COUNTERS 512

struct cuda_ctl {
    int events_count;
    int events_id[PAPI_CUDA_MAX_COUNTERS];
    void * thread_info;  // thread specific info: all gpu user CUcontexts
    void * cupti_ctl;  // pointer to all gpu cuda profiler state
    long long values[PAPI_CUDA_MAX_COUNTERS];
};

event_list_t global_event_names;  // List of event names added by user

papi_vector_t _cuda_pw_vector = {
    .cmp_info = {
        /* default component information (unspecified values are initialized to 0) */
        .name = "cuda_pw",
        .short_name = "cuda_pw",
        .version = "0.1",
        .description = "New CUDA component using PerfWorks API",
        .num_mpx_cntrs = PAPI_CUDA_MAX_COUNTERS,
        .num_cntrs = PAPI_CUDA_MAX_COUNTERS,
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        /* component specific cmp_info initializations */
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .initialized = 0,
    },
    .size = {
        .context = 0,
        .control_state = sizeof(struct cuda_ctl),
    },
    .init_component = cuda_init_component,
    .shutdown_component = cuda_shutdown_component,

    .init_thread = cuda_init_thread,
    .shutdown_thread = cuda_shutdown_thread,

    .ntv_enum_events = cuda_ntv_enum_events,
    .ntv_code_to_name = cuda_ntv_code_to_name,
    .ntv_name_to_code = cuda_ntv_name_to_code,
    .ntv_code_to_descr = cuda_ntv_code_to_descr,

    .init_control_state = cuda_init_control_state,
    .set_domain = cuda_set_domain,
    .update_control_state = cuda_update_control_state,
    .cleanup_eventset = cuda_cleanup_eventset,

    .start = cuda_start,
    .stop = cuda_stop,
    .read = cuda_read,
    .reset = cuda_reset,
};

unsigned _cuda_pw_lock;

static int cuda_init_component(int cidx)
{
    COMPDBG("Entering with component idx: %d\n", cidx);

    _cuda_pw_vector.cmp_info.CmpIdx = cidx;
    _cuda_pw_vector.cmp_info.num_native_events = -1;
    _cuda_pw_vector.cmp_info.num_cntrs = -1;
    _cuda_pw_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    _cuda_pw_vector.cmp_info.initialized = 1;
    _cuda_pw_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    sprintf(_cuda_pw_vector.cmp_info.disabled_reason,
        "Not initialized. Access component events to initialize it.");
    return PAPI_EDELAY_INIT;
}

static int cuda_shutdown_component(void)
{
    COMPDBG("Entering.\n");
    int res = PAPI_OK;
    free_event_name_list(&global_event_names);  // Free the global event names added dynamically

    if (!_cuda_pw_vector.cmp_info.initialized ||
    _cuda_pw_vector.cmp_info.disabled != PAPI_OK)
    return PAPI_OK;

    _cuda_pw_vector.cmp_info.initialized = 0;

    cupti_shutdown();
    // todo: Reverse of private init.
    return res;
}

static int cuda_init_private(void)
{
    int res = PAPI_OK;
    const char *disabled_reason;
    COMPDBG("Entering.\n");
    res = cupti_init(&disabled_reason);
    if (res != PAPI_OK)
    {
        sprintf(_cuda_pw_vector.cmp_info.disabled_reason, disabled_reason);
        _cuda_pw_vector.cmp_info.disabled = res;
        goto fn_exit;
    }

    // Initialize global_event_names array
    res = initialize_dynamic_event_list(&global_event_names);
    if (res != PAPI_OK) {
        goto fn_exit;
    }

    _cuda_pw_vector.cmp_info.disabled = PAPI_OK;
    strcpy(_cuda_pw_vector.cmp_info.disabled_reason, "");

fn_exit:
    return res;
}

static int check_n_initialize(void)
{
    _papi_hwi_lock(COMPONENT_LOCK);
    int res = PAPI_OK;
    if (_cuda_pw_vector.cmp_info.initialized
        && _cuda_pw_vector.cmp_info.disabled == PAPI_EDELAY_INIT
    ) {
        res = cuda_init_private();
    }

    _papi_hwi_unlock(COMPONENT_LOCK);
    return res;
}

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier)
{
    int res = check_n_initialize();
    if (res != PAPI_OK)
        goto fn_exit;

    res = cupti_enumerate_all_events(&global_event_names);
    if (res != PAPI_OK)
        goto fn_exit;

    _cuda_pw_vector.cmp_info.num_cntrs = global_event_names.count;
    _cuda_pw_vector.cmp_info.num_native_events = global_event_names.count;
    switch (modifier) {
        case PAPI_ENUM_FIRST:
            *event_code = 0;
            res = PAPI_OK;
            break;
        case PAPI_ENUM_EVENTS:
            if (global_event_names.count == 0) {
                res = PAPI_ENOEVNT;
            } else if (*event_code < global_event_names.count - 1) {
                *event_code = *event_code + 1;
                res = PAPI_OK;
            } else {
                res = PAPI_ENOEVNT;
            }
            break;
        default:
            res = PAPI_EINVAL;
    }
fn_exit:
    return res;
}

static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code)
{
    int res = check_n_initialize();
    if (res != PAPI_OK)
        return res;
    event_rec_t *evt_rec;
    res = find_event_name(&global_event_names, name, &evt_rec);
    if (res == PAPI_OK) {
        *event_code = evt_rec->evt_code;
        return PAPI_OK;
    }
    else {
        _papi_hwi_lock(COMPONENT_LOCK);
        *event_code = global_event_names.count;
        res = insert_event_record(&global_event_names, name, global_event_names.count, 0);
        _papi_hwi_unlock(COMPONENT_LOCK);
    }

    return PAPI_OK;
}

static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int res = check_n_initialize();
    if (res != PAPI_OK)
        return res;
    if (event_code >= global_event_names.count) {
        return PAPI_ENOEVNT;
    }

    strncpy(name, global_event_names.evts[event_code].name, len);
    return PAPI_OK;
}

static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int __attribute__((unused)) len)
{
    char evt_name[PAPI_2MAX_STR_LEN];
    int res;
    res = check_n_initialize();
    if (res != PAPI_OK)
        goto fn_exit;
    res = cupti_enumerate_all_events(&global_event_names);
    if (res != PAPI_OK)
        goto fn_exit;
    res = cuda_ntv_code_to_name(event_code, evt_name, PAPI_2MAX_STR_LEN);
    if (res != PAPI_OK)
        goto fn_exit;
    res = cupti_get_event_description(evt_name, descr);
fn_exit:
    return res;
}

static int cuda_init_thread(hwd_context_t __attribute__((unused)) *ctx)
{
    return PAPI_OK;
}

static int cuda_shutdown_thread(hwd_context_t __attribute__((unused)) *ctx)
{
    return PAPI_OK;
}

static int cuda_init_control_state(hwd_control_state_t __attribute__((unused)) *ctl)
{
    COMPDBG("Entering.\n");
    return PAPI_OK;
}

static int cuda_set_domain(hwd_control_state_t __attribute__((unused)) *ctrl, int domain)
{
    COMPDBG("Entering\n");
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain))
        return (PAPI_OK);
    else
        return (PAPI_EINVAL);
    return (PAPI_OK);
}

static int cuda_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, __attribute__((unused)) hwd_context_t *ctx
) {
    COMPDBG("Entering with events_count %d.\n", ntv_count);
    int i, res;
    res = check_n_initialize();
    if (res != PAPI_OK)
        return res;
    if (ntv_count == 0)
        goto fn_exit;
    struct cuda_ctl *control = (struct cuda_ctl *) ctl;
    LOCKDBG("Locking.\n");
    _papi_hwi_lock(_cuda_pw_lock);
    LOCKDBG("Locked.\n");
    if (control->thread_info == NULL) {
        res = cupti_thread_info_init(&(control->thread_info));
    }
    control->events_count = ntv_count;

    if (ntv_count > PAPI_CUDA_MAX_COUNTERS) {
        ERRDBG("Too many events added.\n");
        goto fn_fail;
    }
    // Add all event names for each gpu in control state
    for (i=0; i<ntv_count; i++) {
        control->events_id[i] = ntv_info[i].ni_event;  // store the event code added
        ntv_info[i].ni_position = i;  // store the mapping of added event index
    }

    // Validate the added names so far in a temporary context
    void *tmp_context;
    res = cupti_control_create(&global_event_names, control->events_count, control->events_id, &tmp_context, &(control->thread_info));
    if (res != PAPI_OK) {
        cupti_control_destroy(&tmp_context);
        goto fn_exit;
    }
    res = cupti_control_destroy(&tmp_context);

fn_exit:
    LOCKDBG("Unlocking.\n");
    _papi_hwi_unlock(_cuda_pw_lock);
    return res;
fn_fail:
    LOCKDBG("Unlocking.\n");
    _papi_hwi_unlock(_cuda_pw_lock);
    return PAPI_ECMP;
}

static int cuda_cleanup_eventset(hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    struct cuda_ctl *control = (struct cuda_ctl *) ctl;
    int res = PAPI_OK;
    if (control->cupti_ctl)
        res += cupti_control_destroy(&(control->cupti_ctl));
    if (control->thread_info)
        res += cupti_thread_info_free(&(control->thread_info));
    if (res != PAPI_OK)
        return PAPI_ECMP;
    return PAPI_OK;
}

static int cuda_start(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    int res, i;
    LOCKDBG("Locking.\n");
    _papi_hwi_lock(_cuda_pw_lock);
    LOCKDBG("Locked.\n");

    struct cuda_ctl *control = (struct cuda_ctl *) ctl;
    // Set initial counters to zero
    for (i=0; i<control->events_count; i++) {
        control->values[i] = 0;
    }
    res = cupti_control_create(&global_event_names,
                             control->events_count,
                             control->events_id,
                             &(control->cupti_ctl),
                             &(control->thread_info));
    res = cupti_start( &(control->cupti_ctl), &(control->thread_info) );
    LOCKDBG("Unlocking.\n");
    _papi_hwi_unlock(_cuda_pw_lock);
    return res;
}

int cuda_stop(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    LOCKDBG("Locking.\n");
    _papi_hwi_lock(_cuda_pw_lock);
    LOCKDBG("Locked.\n");
    struct cuda_ctl *control = (struct cuda_ctl *) ctl;
    int res;
    res = cupti_stop( &(control->cupti_ctl), &(control->thread_info) );
    if (res != PAPI_OK)
        goto fn_exit;
    res = cupti_control_destroy( &(control->cupti_ctl) );
fn_exit:
    LOCKDBG("Unlocking.\n");
    _papi_hwi_unlock(_cuda_pw_lock);
    return res;
}

static int cuda_read(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl, long long **val, int __attribute__((unused)) flags)
{
    COMPDBG("Entering.\n");
    struct cuda_ctl * control = (struct cuda_ctl *) ctl;
    int res;
    LOCKDBG("Locking.\n");
    _papi_hwi_lock(_cuda_pw_lock);
    LOCKDBG("Locked.\n");
    res = cupti_stop( &(control->cupti_ctl), &(control->thread_info) );
    if (res != PAPI_OK)
        goto fn_exit;
    // First collect the values from the lower layer for last session
    res = cupti_control_read( &(control->cupti_ctl), (long long *) &(control->values) );
    if (res != PAPI_OK)
        goto fn_exit;
    // Then copy the values to the user array `val`
    *val = control->values;

    res = cupti_start( &(control->cupti_ctl), &(control->thread_info) );

fn_exit:
    LOCKDBG("Unlocking.\n");
    _papi_hwi_unlock(_cuda_pw_lock);
    return res;
}

static int cuda_reset(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    struct cuda_ctl * control = (struct cuda_ctl *) ctl;
    int i;
    for (i = 0; i < control->events_count; i++) {
        control->values[i] = 0;
    }
    return cupti_control_reset( &(control->cupti_ctl) );
}
