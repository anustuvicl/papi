/*
 * Contain only functions related to cuda profiler API
 */

#include <dlfcn.h>
#include <papi.h>
#include "papi_memory.h"

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include "lcuda_utils.h"
#include "cupti_profiler.h"
#include "debug_comp.h"

typedef struct byte_array_s         byte_array_t;
typedef struct cuptip_gpu_control_s cuptip_gpu_control_t;
typedef struct cuptip_control_s     cuptip_control_t;
typedef struct list_metrics_s       list_metrics_t;
typedef struct NVPA_MetricsContext  NVPA_MetricsContext;
enum running_e           {False, True};
enum collection_method_e {SpotValue, RunningMin, RunningMax, RunningSum};

static void *dl_nvpw;
static int num_gpus;
static list_metrics_t *avail_events;

static int load_cupti_perf_sym(void);
static int unload_cupti_perf_sym(void);
static int load_nvpw_sym(void);
static int unload_nvpw_sym(void);
static int initialize_cupti_profiler_api(void);
static int deinitialize_cupti_profiler_api(void);
static int initialize_perfworks_api(void);
static int get_chip_name(int dev_num, char* chipName);
static int add_events_per_gpu(event_list_t *event_names, int event_count, int *evt_ids, cuptip_control_t *state);
static int retrieve_metric_details(NVPA_MetricsContext *pMetricsContext, const char *nv_name,
                                   char *description, int *numDep, NVPA_RawMetricRequest **pRMR);
static int get_event_names_rmr(NVPA_MetricsContext* pMetricsContext, cuptip_gpu_control_t *ctl);
static int check_num_passes(struct NVPA_RawMetricsConfig *pRawMetricsConfig, int rmr_count,
                            NVPA_RawMetricRequest *rmr, int *num_pass);
static int control_state_validate(cuptip_control_t *state);
static int get_counter_availability(cuptip_gpu_control_t *ctl);
static int metric_get_config_image(const char *chipName, cuptip_gpu_control_t *ctl);
static int metric_get_counter_data_prefix_image(const char* chipName, cuptip_gpu_control_t *ctl);
static int create_counter_data_image(cuptip_gpu_control_t *ctl);
static int reset_cupti_prof_config_images(cuptip_gpu_control_t *ctl);
static int begin_profiling(cuptip_gpu_control_t *ctl);
static int end_profiling(cuptip_gpu_control_t *ctl);
static int eval_metric_values_per_gpu(NVPA_MetricsContext* pMetricsContext, cuptip_gpu_control_t *ctl);
static int get_metric_values(cuptip_control_t *state);
static int find_same_chipname(int gpu_id);
static int init_all_metrics(void);
static void free_all_enumerated_metrics(void);
static enum collection_method_e get_event_collection_method(const char *evt_name);

NVPA_Status ( *NVPW_GetSupportedChipNamesPtr ) (NVPW_GetSupportedChipNames_Params* params);
NVPA_Status ( *NVPW_CUDA_MetricsContext_CreatePtr ) (NVPW_CUDA_MetricsContext_Create_Params* params);
NVPA_Status ( *NVPW_MetricsContext_DestroyPtr ) (NVPW_MetricsContext_Destroy_Params * params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_BeginPtr ) (NVPW_MetricsContext_GetMetricNames_Begin_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_EndPtr ) (NVPW_MetricsContext_GetMetricNames_End_Params* params);
NVPA_Status ( *NVPW_InitializeHostPtr ) (NVPW_InitializeHost_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_BeginPtr ) (NVPW_MetricsContext_GetMetricProperties_Begin_Params* p);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_EndPtr ) (NVPW_MetricsContext_GetMetricProperties_End_Params* p);
NVPA_Status ( *NVPW_CUDA_RawMetricsConfig_CreatePtr ) (NVPW_CUDA_RawMetricsConfig_Create_Params*);

NVPA_Status ( *NVPW_RawMetricsConfig_DestroyPtr ) (NVPW_RawMetricsConfig_Destroy_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_BeginPassGroupPtr ) (NVPW_RawMetricsConfig_BeginPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_EndPassGroupPtr ) (NVPW_RawMetricsConfig_EndPassGroup_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_AddMetricsPtr ) (NVPW_RawMetricsConfig_AddMetrics_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GenerateConfigImagePtr ) (NVPW_RawMetricsConfig_GenerateConfigImage_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetConfigImagePtr ) (NVPW_RawMetricsConfig_GetConfigImage_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_CreatePtr ) (NVPW_CounterDataBuilder_Create_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_DestroyPtr ) (NVPW_CounterDataBuilder_Destroy_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_AddMetricsPtr ) (NVPW_CounterDataBuilder_AddMetrics_Params* params);
NVPA_Status ( *NVPW_CounterDataBuilder_GetCounterDataPrefixPtr ) (NVPW_CounterDataBuilder_GetCounterDataPrefix_Params* params);
NVPA_Status ( *NVPW_CounterData_GetNumRangesPtr ) (NVPW_CounterData_GetNumRanges_Params* params);
NVPA_Status ( *NVPW_Profiler_CounterData_GetRangeDescriptionsPtr ) (NVPW_Profiler_CounterData_GetRangeDescriptions_Params* params);
NVPA_Status ( *NVPW_MetricsContext_SetCounterDataPtr ) (NVPW_MetricsContext_SetCounterData_Params* params);
NVPA_Status ( *NVPW_MetricsContext_EvaluateToGpuValuesPtr ) (NVPW_MetricsContext_EvaluateToGpuValues_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_GetNumPassesPtr ) (NVPW_RawMetricsConfig_GetNumPasses_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_SetCounterAvailabilityPtr ) (NVPW_RawMetricsConfig_SetCounterAvailability_Params* params);
NVPA_Status ( *NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr ) (NVPW_RawMetricsConfig_IsAddMetricsPossible_Params* params);

NVPA_Status ( *NVPW_MetricsContext_GetCounterNames_BeginPtr ) (NVPW_MetricsContext_GetCounterNames_Begin_Params* pParams);
NVPA_Status ( *NVPW_MetricsContext_GetCounterNames_EndPtr ) (NVPW_MetricsContext_GetCounterNames_End_Params* pParams);

CUptiResult ( *cuptiDeviceGetChipNamePtr ) (CUpti_Device_GetChipName_Params* params);
CUptiResult ( *cuptiProfilerInitializePtr ) (CUpti_Profiler_Initialize_Params* params);
CUptiResult ( *cuptiProfilerDeInitializePtr ) (CUpti_Profiler_DeInitialize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageCalculateSizePtr ) (CUpti_Profiler_CounterDataImage_CalculateSize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageInitializePtr ) (CUpti_Profiler_CounterDataImage_Initialize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr ) (CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params* params);
CUptiResult ( *cuptiProfilerCounterDataImageInitializeScratchBufferPtr ) (CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params* params);
CUptiResult ( *cuptiProfilerBeginSessionPtr ) (CUpti_Profiler_BeginSession_Params* params);
CUptiResult ( *cuptiProfilerSetConfigPtr ) (CUpti_Profiler_SetConfig_Params* params);
CUptiResult ( *cuptiProfilerBeginPassPtr ) (CUpti_Profiler_BeginPass_Params* params);
CUptiResult ( *cuptiProfilerEnableProfilingPtr ) (CUpti_Profiler_EnableProfiling_Params* params);
CUptiResult ( *cuptiProfilerPushRangePtr ) (CUpti_Profiler_PushRange_Params* params);
CUptiResult ( *cuptiProfilerPopRangePtr ) (CUpti_Profiler_PopRange_Params* params);
CUptiResult ( *cuptiProfilerDisableProfilingPtr ) (CUpti_Profiler_DisableProfiling_Params* params);
CUptiResult ( *cuptiProfilerEndPassPtr ) (CUpti_Profiler_EndPass_Params* params);
CUptiResult ( *cuptiProfilerFlushCounterDataPtr ) (CUpti_Profiler_FlushCounterData_Params* params);
CUptiResult ( *cuptiProfilerUnsetConfigPtr ) (CUpti_Profiler_UnsetConfig_Params* params);
CUptiResult ( *cuptiProfilerEndSessionPtr ) (CUpti_Profiler_EndSession_Params* params);
CUptiResult ( *cuptiProfilerGetCounterAvailabilityPtr ) (CUpti_Profiler_GetCounterAvailability_Params* params);
CUptiResult ( *cuptiFinalizePtr ) (void);

#define NVPW_CALL( call, handleerror ) \
    do {  \
        NVPA_Status _status = (call);  \
        LOGCUPTICALL("\t" #call "\033[0m\n");  \
        if (_status != NVPA_STATUS_SUCCESS) {  \
            ANSIRED;  \
            fprintf(stderr, "NVPA Error %d: %s: %d: Error in call to " #call "\n", _status, __FILE__, __LINE__);  \
            ANSIEND;  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

static int load_cupti_perf_sym(void)
{
    COMPDBG("Entering.\n");
    int papiErr = PAPI_OK;
    if (dl_cupti == NULL) {
        ERRDBG("libcupti.so should already be loaded.\n");
        goto fn_fail;
    }

    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");
    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDeInitialize");
    cuptiProfilerCounterDataImageCalculateSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerEndSession");
    cuptiProfilerGetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiProfilerGetCounterAvailability");
    cuptiFinalizePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiFinalize");

fn_exit:
    return papiErr;
fn_fail:
    papiErr = PAPI_ENOSUPP;
    goto fn_exit;
}

static int unload_cupti_perf_sym(void)
{
    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
    cuptiDeviceGetChipNamePtr                                  = NULL;
    cuptiProfilerInitializePtr                                 = NULL;
    cuptiProfilerDeInitializePtr                               = NULL;
    cuptiProfilerCounterDataImageCalculateSizePtr              = NULL;
    cuptiProfilerCounterDataImageInitializePtr                 = NULL;
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = NULL;
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr    = NULL;
    cuptiProfilerBeginSessionPtr                               = NULL;
    cuptiProfilerSetConfigPtr                                  = NULL;
    cuptiProfilerBeginPassPtr                                  = NULL;
    cuptiProfilerEnableProfilingPtr                            = NULL;
    cuptiProfilerPushRangePtr                                  = NULL;
    cuptiProfilerPopRangePtr                                   = NULL;
    cuptiProfilerDisableProfilingPtr                           = NULL;
    cuptiProfilerEndPassPtr                                    = NULL;
    cuptiProfilerFlushCounterDataPtr                           = NULL;
    cuptiProfilerUnsetConfigPtr                                = NULL;
    cuptiProfilerEndSessionPtr                                 = NULL;
    cuptiProfilerGetCounterAvailabilityPtr                     = NULL;
    cuptiFinalizePtr                                           = NULL;
    return PAPI_OK;
}

static int load_nvpw_sym(void)
{
    COMPDBG("Entering.\n");
    char dlname[] = "libnvperf_host.so";
    char *found_files[MAX_FILES];
    int count, i, found = 0;
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root) {
        count = search_files_in_path(dlname, papi_cuda_root, found_files);
        for (i = 0; i < count; i++) {
            dl_nvpw = dlopen(found_files[i], RTLD_NOW | RTLD_GLOBAL);
            if (dl_nvpw) {
                found = 1;
                break;
            }
        }
        for (i = 0; i < count; i++) {
            free(found_files[i]);
        }
    }
    if (!found) {
        dl_nvpw = dlopen(dlname, RTLD_NOW | RTLD_GLOBAL);
        if (!dl_nvpw) {
            ERRDBG("Loading libnvperf_host.so failed.\n");
            goto fn_fail;
        }
    }

    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_GetSupportedChipNames");
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_MetricsContext_Create");
    NVPW_MetricsContext_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_InitializeHost");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetMetricProperties_End");
    NVPW_CUDA_RawMetricsConfig_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetConfigImage");
    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    NVPW_CounterData_GetNumRangesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    NVPW_MetricsContext_SetCounterDataPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_SetCounterAvailability");
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_RawMetricsConfig_IsAddMetricsPossible");
    NVPW_MetricsContext_GetCounterNames_BeginPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetCounterNames_Begin");
    NVPW_MetricsContext_GetCounterNames_EndPtr = DLSYM_AND_CHECK(dl_nvpw, "NVPW_MetricsContext_GetCounterNames_End");

    Dl_info info;
    dladdr(NVPW_GetSupportedChipNamesPtr, &info);
    LOGDBG("NVPW library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_ENOSUPP;
}

static int unload_nvpw_sym(void)
{
    if (dl_nvpw) {
        dlclose(dl_nvpw);
        dl_nvpw = NULL;
    }
    NVPW_GetSupportedChipNamesPtr                     = NULL;
    NVPW_CUDA_MetricsContext_CreatePtr                = NULL;
    NVPW_MetricsContext_DestroyPtr                    = NULL;
    NVPW_MetricsContext_GetMetricNames_BeginPtr       = NULL;
    NVPW_MetricsContext_GetMetricNames_EndPtr         = NULL;
    NVPW_InitializeHostPtr                            = NULL;
    NVPW_MetricsContext_GetMetricProperties_BeginPtr  = NULL;
    NVPW_MetricsContext_GetMetricProperties_EndPtr    = NULL;
    NVPW_CUDA_RawMetricsConfig_CreatePtr              = NULL;
    NVPW_RawMetricsConfig_DestroyPtr                  = NULL;
    NVPW_RawMetricsConfig_BeginPassGroupPtr           = NULL;
    NVPW_RawMetricsConfig_EndPassGroupPtr             = NULL;
    NVPW_RawMetricsConfig_AddMetricsPtr               = NULL;
    NVPW_RawMetricsConfig_GenerateConfigImagePtr      = NULL;
    NVPW_RawMetricsConfig_GetConfigImagePtr           = NULL;
    NVPW_CounterDataBuilder_CreatePtr                 = NULL;
    NVPW_CounterDataBuilder_DestroyPtr                = NULL;
    NVPW_CounterDataBuilder_AddMetricsPtr             = NULL;
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr   = NULL;
    NVPW_CounterData_GetNumRangesPtr                  = NULL;
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = NULL;
    NVPW_MetricsContext_SetCounterDataPtr             = NULL;
    NVPW_MetricsContext_EvaluateToGpuValuesPtr        = NULL;
    NVPW_RawMetricsConfig_GetNumPassesPtr             = NULL;
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr   = NULL;
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr     = NULL;
    NVPW_MetricsContext_GetCounterNames_BeginPtr      = NULL;
    NVPW_MetricsContext_GetCounterNames_EndPtr        = NULL;
    return PAPI_OK;
}

static int initialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE, NULL };
    if (cuptiProfilerInitializePtr(&profilerInitializeParams) == CUPTI_SUCCESS)
        return PAPI_OK;
    else
        return PAPI_ESYS;  // or something else?
}

static int deinitialize_cupti_profiler_api(void)
{
    COMPDBG("Entering.\n");
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE, NULL };
    if (cuptiProfilerDeInitializePtr(&profilerDeInitializeParams) == CUPTI_SUCCESS)
        return PAPI_OK;
    else
        return PAPI_ESYS;
}

static int initialize_perfworks_api(void)
{
    COMPDBG("Entering.\n");
    NVPW_InitializeHost_Params perfInitHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE, NULL };
    if (NVPW_InitializeHostPtr(&perfInitHostParams) == NVPA_STATUS_SUCCESS)
        return PAPI_OK;
    else
        return PAPI_ESYS;
    // NVPW_InitializeTarget_Params perfInitTargetParams = {NVPW_InitializeTarget_Params_STRUCT_SIZE };
    // NVPW_CALL(NVPW_InitializeTarget(&perfInitTargetParams));
    return PAPI_OK;
}

static int get_chip_name(int dev_num, char* chipName)
{
    CUpti_Device_GetChipName_Params getChipName = {
        .structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .deviceIndex = 0
    };
    getChipName.deviceIndex = dev_num;
    CUPTI_CALL( cuptiDeviceGetChipNamePtr(&getChipName), return PAPI_ENOSUPP );
    strcpy(chipName, getChipName.pChipName);
    return PAPI_OK;
}

struct byte_array_s {
    int size;
    uint8_t *data;
};

struct cuptip_gpu_control_s {
    int gpu_id;
    event_list_t event_names;
    int rmr_count;
    NVPA_RawMetricRequest *rmr;
    byte_array_t counterDataImagePrefix;
    byte_array_t configImage;
    byte_array_t counterDataImage;
    byte_array_t counterDataScratchBuffer;
    byte_array_t counterAvailabilityImage;
};

struct cuptip_control_s {
    cuptip_gpu_control_t *ctl;
    int read_count;
    enum running_e running;
};

static int add_events_per_gpu(event_list_t *event_names, int event_count, int *evt_ids, cuptip_control_t *state)
{
    COMPDBG("Entering.\n");
    int i, gpu_id, res = PAPI_OK;
    char nvName[PAPI_MAX_STR_LEN];
    for (i=0; i < num_gpus; i++) {
        res = initialize_dynamic_event_list( &(state->ctl[i].event_names) );
    }
    for (i = 0; i < event_count; i++) {
        res = tokenize_event_name(event_names->evts[evt_ids[i]].name, (char*) &nvName, &gpu_id);
        if (res != PAPI_OK)
            goto fn_exit;
        if (gpu_id < 0 || gpu_id > num_gpus) {
            res = PAPI_EINVAL;
            goto fn_exit;
        }
        insert_event_record(&(state->ctl[gpu_id].event_names),
                            event_names->evts[evt_ids[i]].name,
                            event_names->evts[evt_ids[i]].evt_code,
                            i);
        LOGDBG("Adding event gpu %d name %s with code %d at pos %d\n", gpu_id, event_names->evts[evt_ids[i]].name,
                                                                       event_names->evts[evt_ids[i]].evt_code, i);

    }
fn_exit:
    return res;
}

static int retrieve_metric_details(NVPA_MetricsContext *pMetricsContext, const char *nv_name,
                                   char *description, int *numDep, NVPA_RawMetricRequest **pRMR)
{
    COMPDBG("Entering.\n");
    int num_dep, i, len;
    NVPA_RawMetricRequest *rmr;
    NVPA_Status nvpa_err;

    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = {
        .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
        .pMetricName = nv_name,
    };
    nvpa_err = NVPW_MetricsContext_GetMetricProperties_BeginPtr(&getMetricPropertiesBeginParams);

    if (nvpa_err != NVPA_STATUS_SUCCESS || getMetricPropertiesBeginParams.ppRawMetricDependencies == NULL) {
        strcpy(description, "Could not get description.");
        return PAPI_EINVAL;
    }

    for (num_dep=0; getMetricPropertiesBeginParams.ppRawMetricDependencies[num_dep] != NULL; num_dep++){;}

    rmr = (NVPA_RawMetricRequest *) papi_calloc(num_dep, sizeof(NVPA_RawMetricRequest));
    if (rmr == NULL) {
        return PAPI_ENOMEM;
    }

    for (i=0; i<num_dep; i++) {
        rmr[i].pMetricName = strdup(getMetricPropertiesBeginParams.ppRawMetricDependencies[i]);
        rmr[i].isolated = 1;
        rmr[i].keepInstances = 1;
        rmr[i].structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
    }

    len = snprintf(description, PAPI_2MAX_STR_LEN, "%s. Units=(%s)",
                getMetricPropertiesBeginParams.pDescription,
                getMetricPropertiesBeginParams.pDimUnits);
    if (len > PAPI_2MAX_STR_LEN) {
        ERRDBG("String formatting exceeded max string length.\n");
        return PAPI_ENOMEM;
    }
    *numDep = num_dep;
    *pRMR = rmr;
    NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = {
        .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
    };
    NVPW_CALL(NVPW_MetricsContext_GetMetricProperties_EndPtr(&getMetricPropertiesEndParams), return PAPI_ENOSUPP);
    return PAPI_OK;
}

static int get_event_names_rmr(NVPA_MetricsContext* pMetricsContext, cuptip_gpu_control_t *ctl)
{
    COMPDBG("Entering.\n");
    int res = PAPI_OK;
    NVPA_RawMetricRequest *all_rmr=NULL;
    int count_raw_metrics = 0;
    unsigned int i;
    int j, k, num_dep;
    NVPA_RawMetricRequest *temp;
    char nv_name[PAPI_MAX_STR_LEN]; int gpuid;

    for (i=0; i < ctl->event_names.count; i++) {
        res = tokenize_event_name(ctl->event_names.evts[i].name, (char *) &nv_name, &gpuid);

        res = retrieve_metric_details(pMetricsContext, nv_name, ctl->event_names.evts[i].desc, &num_dep, &temp);
        if (res != PAPI_OK) {
            res = PAPI_ENOEVNT;
            goto fn_exit;
        }

        // Collect and build array of all raw metric requests
        all_rmr = (NVPA_RawMetricRequest *) papi_realloc(all_rmr, (count_raw_metrics + num_dep) * sizeof(NVPA_RawMetricRequest));
        if (all_rmr == NULL) {
            res = PAPI_ENOMEM;
            goto fn_exit;
        }
        for (j = 0; j < num_dep; j++) {
            k = j + count_raw_metrics;
            all_rmr[k].structSize = temp[j].structSize;
            all_rmr[k].pPriv = NULL;
            all_rmr[k].pMetricName = strdup(temp[j].pMetricName);
            all_rmr[k].keepInstances = 1;
            all_rmr[k].isolated = 1;
            papi_free((void *) temp[j].pMetricName);
        }
        count_raw_metrics += num_dep;
        papi_free(temp);
    }
    ctl->rmr = all_rmr;
    ctl->rmr_count = count_raw_metrics;
fn_exit:
    return res;
}

static int check_num_passes(struct NVPA_RawMetricsConfig *pRawMetricsConfig, int rmr_count, NVPA_RawMetricRequest *rmr, int *num_pass)
{
    COMPDBG("Entering.\n");
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
        .maxPassCount = 1,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
        .pRawMetricRequests = rmr,
        .numMetricRequests = rmr_count,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_GetNumPasses_Params rawMetricsConfigGetNumPassesParams = {
        .structSize = NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL( NVPW_RawMetricsConfig_GetNumPassesPtr(&rawMetricsConfigGetNumPassesParams), return PAPI_ENOSUPP );

    int numNestingLevels = 1, numIsolatedPasses, numPipelinedPasses;
    numIsolatedPasses  = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;

    *num_pass = numPipelinedPasses + numIsolatedPasses * numNestingLevels;

    if (*num_pass > 1) {
        ERRDBG("Metrics requested requires multiple passes to profile.\n");
        return PAPI_EMULPASS;
    }

    return PAPI_OK;
}

static int control_state_validate(cuptip_control_t *state)
{
    COMPDBG("Entering.\n");
    int gpu_id, res = PAPI_OK, passes;
    char chip_name[32];

    for (gpu_id=0; gpu_id < num_gpus; gpu_id++) {
        if (state->ctl[gpu_id].event_names.count == 0)
            continue;
        res = get_chip_name(gpu_id, chip_name);
        if (res != PAPI_OK) {
            ERRDBG("Failed to get chipname.\n");
            goto fn_exit;
        }

        NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
            .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pChipName = chip_name,
        };
        NVPW_CALL( NVPW_CUDA_MetricsContext_CreatePtr(&metricsContextCreateParams), return PAPI_ENOSUPP);
        if (metricsContextCreateParams.pMetricsContext == NULL) {
            ERRDBG("Failed to create metrics context!\n");
            res = PAPI_ECMP;
            goto fn_exit;
        }

        res = get_event_names_rmr(metricsContextCreateParams.pMetricsContext, &(state->ctl[gpu_id]));

        NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
            .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = metricsContextCreateParams.pMetricsContext,
        };
        NVPW_CALL(NVPW_MetricsContext_DestroyPtr(&metricsContextDestroyParams), return PAPI_ENOSUPP);

        if (res != PAPI_OK) {
            goto fn_exit;
        }
        NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
            .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
            .pChipName = chip_name,
        };
        NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), return PAPI_ENOSUPP );

        res = check_num_passes(nvpw_metricsConfigCreateParams.pRawMetricsConfig,
                               state->ctl[gpu_id].rmr_count, state->ctl[gpu_id].rmr, &passes);

        NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
            .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        };
        NVPW_CALL(NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), return PAPI_ENOSUPP);
        if (res != PAPI_OK) {
            goto fn_exit;
        }
    }
fn_exit:
    return res;
}

static int get_counter_availability(cuptip_gpu_control_t *ctl)
{
    int res;
    // Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
        .structSize = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pCounterAvailabilityImage = NULL,
    };
    res = cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams);
    if (res != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get size.\n", res);
        return PAPI_ENOSUPP;
    }
    // Allocate sized counterAvailabilityImage
    ctl->counterAvailabilityImage.size = getCounterAvailabilityParams.counterAvailabilityImageSize;
    ctl->counterAvailabilityImage.data = (uint8_t *) papi_malloc(ctl->counterAvailabilityImage.size);

    // Initialize counterAvailabilityImage
    getCounterAvailabilityParams.pCounterAvailabilityImage = ctl->counterAvailabilityImage.data;
    res = cuptiProfilerGetCounterAvailabilityPtr(&getCounterAvailabilityParams);
    if (res != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get bytes.\n", res);
        return PAPI_ENOSUPP;
    }
    return PAPI_OK;
}

// Adapted from `metric.cpp` -------------------------
static int metric_get_config_image(const char *chipName, cuptip_gpu_control_t *ctl)
{
    COMPDBG("Entering.\n");
    // LOGDBG("RMRcount = %d, metric = %s\n", ctl->rmr_count, ctl->rmr[0].pMetricName);
    NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
        .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
        .pChipName = chipName,
    };
    NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), return PAPI_ENOSUPP);

    if( ctl->counterAvailabilityImage.data != NULL)
    {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setCounterAvailabilityParams = {
            .structSize = NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
            .pCounterAvailabilityImage = ctl->counterAvailabilityImage.data,
        };
        NVPW_CALL(NVPW_RawMetricsConfig_SetCounterAvailabilityPtr(&setCounterAvailabilityParams), return PAPI_ENOSUPP);
    };

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .maxPassCount = 1,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_BeginPassGroupPtr(&beginPassGroupParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .pRawMetricRequests = ctl->rmr,
        .numMetricRequests = ctl->rmr_count,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_AddMetricsPtr(&addMetricsParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        .structSize = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_EndPassGroupPtr(&endPassGroupParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_GenerateConfigImagePtr(&generateConfigImageParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
        .structSize = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), return PAPI_ENOSUPP);

    ctl->configImage.size = getConfigImageParams.bytesCopied;
    ctl->configImage.data = (uint8_t *) papi_calloc(ctl->configImage.size, sizeof(uint8_t));
    if (ctl->configImage.data == NULL) {
        ERRDBG("calloc ctl->configImage.data failed!");
        return PAPI_ENOMEM;
    }

    getConfigImageParams.bytesAllocated = ctl->configImage.size;
    getConfigImageParams.pBuffer = ctl->configImage.data;
    NVPW_CALL(NVPW_RawMetricsConfig_GetConfigImagePtr(&getConfigImageParams), return PAPI_ENOSUPP);

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), return PAPI_ENOSUPP);

    return PAPI_OK;
}

static int metric_get_counter_data_prefix_image(const char* chipName, cuptip_gpu_control_t *ctl)
{
    COMPDBG("Entering.\n");
    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
        .structSize = NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pChipName = chipName,
    };
    NVPW_CALL(NVPW_CounterDataBuilder_CreatePtr(&counterDataBuilderCreateParams), return PAPI_ENOSUPP);

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
        .structSize = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .pRawMetricRequests = ctl->rmr,
        .numMetricRequests = ctl->rmr_count,
    };
    NVPW_CALL(NVPW_CounterDataBuilder_AddMetricsPtr(&addMetricsParams), return PAPI_ENOSUPP);

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {
        .structSize = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), return PAPI_ENOSUPP);

    ctl->counterDataImagePrefix.size = getCounterDataPrefixParams.bytesCopied;
    ctl->counterDataImagePrefix.data = (uint8_t *) papi_calloc(ctl->counterDataImagePrefix.size, sizeof(uint8_t));
    if (ctl->counterDataImagePrefix.data == NULL) {
        ERRDBG("calloc ctl->counterDataImagePrefix.data failed!");
        return PAPI_ENOMEM;
    }

    getCounterDataPrefixParams.bytesAllocated = ctl->counterDataImagePrefix.size;
    getCounterDataPrefixParams.pBuffer = ctl->counterDataImagePrefix.data;
    NVPW_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefixPtr(&getCounterDataPrefixParams), return PAPI_ENOSUPP);

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
        .structSize = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
    };
    NVPW_CALL(NVPW_CounterDataBuilder_DestroyPtr(&counterDataBuilderDestroyParams), return PAPI_ENOSUPP);

    return PAPI_OK;
}

// Adapted from 11.2/cupti-samples/userrange_profiling/simplecuda.cu:CreateCounterDataImage
static int create_counter_data_image(cuptip_gpu_control_t *ctl)
{
    COMPDBG("Entering.\n");
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions = {
        .structSize = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pPriv = NULL,
        .pCounterDataPrefix = ctl->counterDataImagePrefix.data,
        .counterDataPrefixSize = ctl->counterDataImagePrefix.size,
        .maxNumRanges = 1,
        .maxNumRangeTreeNodes = 1,
        .maxRangeNameLength = 64,
    };

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &counterDataImageOptions,
    };
    CUPTI_CALL(cuptiProfilerCounterDataImageCalculateSizePtr(&calculateSizeParams), return PAPI_ENOSUPP);

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &counterDataImageOptions,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
    };

    ctl->counterDataImage.size = calculateSizeParams.counterDataImageSize;
    ctl->counterDataImage.data = (uint8_t *) papi_calloc(ctl->counterDataImage.size, sizeof(uint8_t));
    if (ctl->counterDataImage.data == NULL) {
        ERRDBG("calloc ctl->counterDataImage.data failed!\n");
        return PAPI_ENOMEM;
    }

    initializeParams.pCounterDataImage = ctl->counterDataImage.data;
    CUPTI_CALL(cuptiProfilerCounterDataImageInitializePtr(&initializeParams), return PAPI_ENOSUPP);

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {
        .structSize = CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = initializeParams.pCounterDataImage,
    };
    CUPTI_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr(&scratchBufferSizeParams), return PAPI_ENOSUPP);

    ctl->counterDataScratchBuffer.size = scratchBufferSizeParams.counterDataScratchBufferSize;
    ctl->counterDataScratchBuffer.data = (uint8_t *) papi_calloc(ctl->counterDataScratchBuffer.size, sizeof(uint8_t));
    if (ctl->counterDataScratchBuffer.data == NULL) {
        ERRDBG("calloc ctl->counterDataScratchBuffer.data failed!\n");
        return PAPI_ENOMEM;
    }

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {
        .structSize = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = initializeParams.pCounterDataImage,
        .counterDataScratchBufferSize = ctl->counterDataScratchBuffer.size,
        .pCounterDataScratchBuffer = ctl->counterDataScratchBuffer.data,
    };
    CUPTI_CALL(cuptiProfilerCounterDataImageInitializeScratchBufferPtr(&initScratchBufferParams), return PAPI_ENOSUPP);

    return PAPI_OK;
}

static int reset_cupti_prof_config_images(cuptip_gpu_control_t *ctl)
{
    COMPDBG("Entering.\n");
    papi_free(ctl->counterDataImagePrefix.data);
    papi_free(ctl->configImage.data);
    papi_free(ctl->counterDataImage.data);
    papi_free(ctl->counterDataScratchBuffer.data);
    papi_free(ctl->counterAvailabilityImage.data);
    return PAPI_OK;
}

static int begin_profiling(cuptip_gpu_control_t *ctl)
{
    COMPDBG("Entering.\n");
    byte_array_t *configImage = &(ctl->configImage);
    byte_array_t *counterDataScratchBuffer = &(ctl->counterDataScratchBuffer);
    byte_array_t *counterDataImage = &(ctl->counterDataImage);

    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        .structSize = CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .counterDataImageSize = counterDataImage->size,
        .pCounterDataImage = counterDataImage->data,
        .counterDataScratchBufferSize = counterDataScratchBuffer->size,
        .pCounterDataScratchBuffer = counterDataScratchBuffer->data,
        .range = CUPTI_UserRange,
        .replayMode = CUPTI_UserReplay,
        .maxRangesPerPass = 1,
        .maxLaunchesPerPass = 1,
    };
    CUPTI_CALL(cuptiProfilerBeginSessionPtr(&beginSessionParams), return PAPI_ENOSUPP);

    CUpti_Profiler_SetConfig_Params setConfigParams = {
        .structSize = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pConfig = configImage->data,
        .configSize = configImage->size,
        .minNestingLevel = 1,
        .numNestingLevels = 1,
        .passIndex = 0,
        .targetNestingLevel = 1,
    };
    CUPTI_CALL(cuptiProfilerSetConfigPtr(&setConfigParams), return PAPI_ENOSUPP);

    /* User takes the resposiblity of replaying the kernel launches */
    CUpti_Profiler_BeginPass_Params beginPassParams = {
        .structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerBeginPassPtr(&beginPassParams), return PAPI_ENOSUPP);

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        .structSize = CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerEnableProfilingPtr(&enableProfilingParams), return PAPI_ENOSUPP);

    char rangeName[64];
    sprintf(rangeName, "PAPI_Range_%d", ctl->gpu_id);
    CUpti_Profiler_PushRange_Params pushRangeParams = {
        .structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
        .pRangeName = (const char*) &rangeName,
        .rangeNameLength = 100,
    };
    CUPTI_CALL(cuptiProfilerPushRangePtr(&pushRangeParams), return PAPI_ENOSUPP);

    return PAPI_OK;
}

static int end_profiling(cuptip_gpu_control_t *ctl)
{

    COMPDBG("EndProfiling. dev = %d\n", ctl->gpu_id);
    (void) ctl;

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        .structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerDisableProfilingPtr(&disableProfilingParams), return PAPI_ENOSUPP );

    CUpti_Profiler_PopRange_Params popRangeParams = {
        .structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerPopRangePtr(&popRangeParams), return PAPI_ENOSUPP );

    CUpti_Profiler_EndPass_Params endPassParams = {
        .structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerEndPassPtr(&endPassParams), return PAPI_ENOSUPP );

    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        .structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerFlushCounterDataPtr(&flushCounterDataParams), return PAPI_ENOSUPP );

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
        .structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerUnsetConfigPtr(&unsetConfigParams), return PAPI_ENOSUPP );

    CUpti_Profiler_EndSession_Params endSessionParams = {
        .structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .ctx = NULL,
    };
    CUPTI_CALL(cuptiProfilerEndSessionPtr(&endSessionParams), return PAPI_ENOSUPP );

    return PAPI_OK;
}

// Adapted from Eval.cpp
static int eval_metric_values_per_gpu(NVPA_MetricsContext* pMetricsContext, cuptip_gpu_control_t *ctl)
{
    COMPDBG("eval_metric_values. dev = %d\n", ctl->gpu_id);
    if (!ctl->counterDataImage.size) {
        ERRDBG("Counter Data Image is empty!\n");
        return PAPI_EINVAL;
    }
    int i, res=PAPI_OK;
    int numMetrics = ctl->event_names.count;

    // Create array of added metric names for this gpu
    char **metricNames;
    int dummy;
    metricNames = (char**) papi_calloc(numMetrics, sizeof(char *));
    if (metricNames == NULL) {
        ERRDBG("calloc metricNames failed.\n");
        return PAPI_ENOMEM;
    }
    for (i=0; i<numMetrics; i++) {
        res = tokenize_event_name(ctl->event_names.evts[i].name, ctl->event_names.evts[i].desc, &dummy);
        if (res != PAPI_OK)
            goto fn_exit;
        metricNames[i] = (char *) &(ctl->event_names.evts[i].desc);
        LOGDBG("Setting metric name %s\n", metricNames[i]);
    }

    double* gpuValues;

    gpuValues = (double*) papi_malloc(numMetrics * sizeof(double));

    NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
        .structSize = NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
        .pCounterDataImage = ctl->counterDataImage.data,
        .rangeIndex = 0,
        .isolated = 1,
    };
    NVPW_CALL(NVPW_MetricsContext_SetCounterDataPtr(&setCounterDataParams), return PAPI_ENOSUPP);
    NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
        .structSize = NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .pMetricsContext = pMetricsContext,
        .numMetrics = numMetrics,
        .ppMetricNames = (const char* const*) metricNames,
        .pMetricValues = gpuValues,
    };
    NVPW_CALL(NVPW_MetricsContext_EvaluateToGpuValuesPtr(&evalToGpuParams), return PAPI_ENOSUPP);
    papi_free(metricNames);
    // char rangeName[100];
    for (i=0; i < (int) ctl->event_names.count; i++) {
        ctl->event_names.evts[i].value = gpuValues[i];
    }
    papi_free(gpuValues);
fn_exit:
    return res;
}

static int get_metric_values(cuptip_control_t *state)
{
    COMPDBG("Entering.\n");
    int gpu_id, res = PAPI_OK, i;
    char chip_name[32];
    cuptip_gpu_control_t *ctl;
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        ctl = &(state->ctl[gpu_id]);
        if (ctl->event_names.count == 0)
            continue;
        res = get_chip_name(gpu_id, chip_name);
        if (res != PAPI_OK) {
            ERRDBG("Failed to get chipname.\n");
            goto fn_exit;
        }

        NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
            .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pChipName = chip_name,
        };
        NVPW_CALL( NVPW_CUDA_MetricsContext_CreatePtr(&metricsContextCreateParams), return PAPI_ENOSUPP);
        if (metricsContextCreateParams.pMetricsContext == NULL) {
            ERRDBG("Failed to create metrics context!\n");
            res = PAPI_ECMP;
            goto fn_exit;
        }

        res = eval_metric_values_per_gpu(metricsContextCreateParams.pMetricsContext, ctl);
        LOGDBG("Measured values from gpu %d\n", ctl->gpu_id);
        for (i=0; i < (int) ctl->event_names.count; i++) {
            LOGDBG("%s\t%lf\n", ctl->event_names.evts[i].name, ctl->event_names.evts[i].value);  // check nv_name here
        }

        NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
            .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = metricsContextCreateParams.pMetricsContext,
        };
        NVPW_CALL(NVPW_MetricsContext_DestroyPtr(&metricsContextDestroyParams), return PAPI_ENOSUPP);

    }
fn_exit:
    return res;
}

// List metrics API
struct list_metrics_s {
    char chip_name[32];
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams;
    int num_metrics;
    event_list_t *nv_metrics;
};

static int find_same_chipname(int gpu_id)
{
    int i;
    for (i=0; i<gpu_id; i++) {
        if (!strcmp(avail_events[gpu_id].chip_name, avail_events[i].chip_name)) {
            return i;
        }
    }
    return -1;  // indicates not found
}

static int init_all_metrics(void)
{
    int gpu_id, res;
    avail_events = (list_metrics_t *) papi_calloc(num_gpus, sizeof(list_metrics_t));
    if (avail_events == NULL)
        return PAPI_ENOMEM;
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        res = get_chip_name(gpu_id, avail_events[gpu_id].chip_name);
        if (res != PAPI_OK)
            return res;
    }
    return PAPI_OK;
}

int cuptip_enumerate_all_metric_names(event_list_t *all_evt_names)
{
    int gpu_id, i, found, listsubmetrics=1, res;
    if (avail_events[0].nv_metrics != NULL)  // Already eumerated for 1st device? Then exit...
        goto fn_exit;
    STOPWATCH;
    TICK;
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        LOGDBG("Getting metric names for gpu %d\n", gpu_id);
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            avail_events[gpu_id].num_metrics = avail_events[found].num_metrics;
            avail_events[gpu_id].nv_metrics = avail_events[found].nv_metrics;
            continue;
        }
        // If same chip_name not found, get all the details for this gpu

        avail_events[gpu_id].metricsContextCreateParams = (NVPW_CUDA_MetricsContext_Create_Params) {
            .structSize = NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pChipName = avail_events[gpu_id].chip_name,
        };
        NVPW_CALL(NVPW_CUDA_MetricsContext_CreatePtr(&(avail_events[gpu_id].metricsContextCreateParams)),
            return PAPI_ENOSUPP);

        NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = {
            .structSize = NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = avail_events[gpu_id].metricsContextCreateParams.pMetricsContext,
            .hidePeakSubMetrics = !listsubmetrics,
            .hidePerCycleSubMetrics = !listsubmetrics,
            .hidePctOfPeakSubMetrics = !listsubmetrics,
        };
        NVPW_CALL(NVPW_MetricsContext_GetMetricNames_BeginPtr(&getMetricNameBeginParams), return PAPI_ENOSUPP);

        avail_events[gpu_id].num_metrics = getMetricNameBeginParams.numMetrics;
        avail_events[gpu_id].nv_metrics = (event_list_t *) malloc(sizeof(event_list_t));
        if (avail_events[gpu_id].nv_metrics == NULL) {
            return PAPI_ENOMEM;
        }
        res = initialize_dynamic_event_list_size(avail_events[gpu_id].nv_metrics, avail_events[gpu_id].num_metrics);
        if (res != PAPI_OK)
            goto fn_exit;
        for (i=0; i<avail_events[gpu_id].num_metrics; i++) {
            res = insert_event_record(avail_events[gpu_id].nv_metrics,
                                      getMetricNameBeginParams.ppMetricNames[i],
                                      i, 0);
        }

        NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = {
            .structSize = NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = avail_events[gpu_id].metricsContextCreateParams.pMetricsContext,
        };
        NVPW_CALL(NVPW_MetricsContext_GetMetricNames_EndPtr((NVPW_MetricsContext_GetMetricNames_End_Params *) &getMetricNameEndParams), return PAPI_ENOSUPP);

    }
    TOCK;
    TIMEDBG("Time to get all metric names =");
    TICK;
    char evt_name[PAPI_2MAX_STR_LEN];
    event_rec_t *find=NULL;
    int len;
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        for (i=0; i<avail_events[gpu_id].num_metrics; i++) {
            len = snprintf(evt_name, PAPI_2MAX_STR_LEN, "%s:device=%d", avail_events[gpu_id].nv_metrics->evts[i].name, gpu_id);
            if (len > PAPI_2MAX_STR_LEN) {
                ERRDBG("String formatting exceeded maximum length.\n");
                return PAPI_ENOMEM;
            }
            if (find_event_name(all_evt_names, evt_name, &find) == PAPI_ENOEVNT) {
                res = insert_event_record(all_evt_names, evt_name, i, 0);
            }
            // LOGDBG("Name:\t\t%s\nDescription:\t%s\n\n",
            //    all_evt_names->evts[all_evt_names->count-1].name, avail_events[gpu_id].nv_metrics->evts[i].desc);
        }
    }
    TOCK;
    TIMEDBG("Time to transfer all metric names to component =");
    LOGDBG("Total metric names for %d gpus = %d\n", num_gpus, all_evt_names->count);
fn_exit:
    return PAPI_OK;
}

int cuptip_get_event_description(const char *evt_name, char *description)
{
    int res, numdep, gpu_id, passes;
    char nv_name[PAPI_MAX_STR_LEN];
    event_rec_t *evt_rec=NULL;
    NVPA_RawMetricRequest *temp;
    res = tokenize_event_name(evt_name, nv_name, &gpu_id);
    if (res != PAPI_OK)
        goto fn_exit;
    res = find_event_name(avail_events[gpu_id].nv_metrics, nv_name, &evt_rec);
    if (res != PAPI_OK) {
        ERRDBG("Event name not found in avail_events array.\n");
        goto fn_exit;
    }
    char *desc = evt_rec->desc;
    if (desc[0] == '\0') {
        res = retrieve_metric_details(avail_events[gpu_id].metricsContextCreateParams.pMetricsContext,
                                      nv_name, desc, &numdep, &temp);
        if (res == PAPI_OK) {
            NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
                .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
                .pChipName = avail_events[gpu_id].chip_name,
            };
            NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), return PAPI_ENOSUPP );

            res = check_num_passes(nvpw_metricsConfigCreateParams.pRawMetricsConfig,
                                   numdep, temp, &passes);

            NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
                .structSize = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pRawMetricsConfig = nvpw_metricsConfigCreateParams.pRawMetricsConfig,
            };
            NVPW_CALL(NVPW_RawMetricsConfig_DestroyPtr((NVPW_RawMetricsConfig_Destroy_Params *) &rawMetricsConfigDestroyParams), return PAPI_ENOSUPP);

            snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " Numpass=%d", passes);
            if (passes > 1)
                snprintf(desc + strlen(desc), PAPI_2MAX_STR_LEN - strlen(desc), " (multi-pass not supported)");
        }
        papi_free(temp);
    }
    strcpy(description, desc);
fn_exit:
    return res;
}

static void free_all_enumerated_metrics(void)
{
    COMPDBG("Entering.\n");
    int gpu_id, found;
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams;
    if (avail_events == NULL)
        return;
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        found = find_same_chipname(gpu_id);
        if (found > -1) {
            avail_events[gpu_id].num_metrics = 0;
            avail_events[gpu_id].nv_metrics = NULL;
            continue;
        }
        if (avail_events[gpu_id].metricsContextCreateParams.pMetricsContext) {
            metricsContextDestroyParams = (NVPW_MetricsContext_Destroy_Params) {
                .structSize = NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
                .pPriv = NULL,
                .pMetricsContext = avail_events[gpu_id].metricsContextCreateParams.pMetricsContext,
            };
            NVPW_MetricsContext_DestroyPtr(&metricsContextDestroyParams);
        }
        if (avail_events[gpu_id].nv_metrics)
            free_event_name_list(avail_events[gpu_id].nv_metrics);
    }
    free(avail_events);
    avail_events = NULL;
}

// CUPTI Profiler component API functions
int cuptip_init(const char **pdisabled_reason)
{
    COMPDBG("Entering.\n");
    int retval = PAPI_OK;

    retval = load_cupti_perf_sym();
    retval += load_nvpw_sym();
    if (retval != PAPI_OK) {
        *pdisabled_reason = "Unable to load CUDA library functions.";
        goto fn_fail;
    }
    num_gpus = get_device_count();
    if (num_gpus <= 0) {
        *pdisabled_reason = "No GPUs found on system.";
        goto fn_fail;
    }

    retval = initialize_cupti_profiler_api();
    retval += initialize_perfworks_api();
    if (retval != PAPI_OK) {
        *pdisabled_reason = "Unable to initialize CUPTI profiler libraries.";
        goto fn_fail;
    }
    retval = init_all_metrics();
    if (retval != PAPI_OK)
        goto fn_fail;
    retval = cuInitPtr(0);
    if (retval != CUDA_SUCCESS) {
        *pdisabled_reason = "Failed to initialize CUDA driver API.";
        goto fn_fail;
    }
    return PAPI_OK;
fn_fail:
    return PAPI_ECMP;
}

int cuptip_control_create(event_list_t *all_event_names, int event_count, int *evt_ids, void **pctl, void **pcu_ctx)
{
    COMPDBG("Entering.\n");
    int res = PAPI_OK, gpu_id;
    // Allocate profiler control state
    cuptip_control_t *state = (cuptip_control_t *) papi_calloc (1, sizeof(cuptip_control_t));
    if (state == NULL) {
        return PAPI_ENOMEM;
    }
    // Allocate array of profiler states for each gpu
    state->ctl = (cuptip_gpu_control_t *) papi_calloc(num_gpus, sizeof(cuptip_gpu_control_t));
    if (state->ctl == NULL) {
        return PAPI_ENOMEM;
    }
    // Set gpu_id for each profiler state
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++)
        state->ctl[gpu_id].gpu_id = gpu_id;

    // Register the user created cuda context for the current gpu if not already known
    CUcontext *cu_ctx = (CUcontext *) (*pcu_ctx);
    CUcontext tempCtx;
    res = cudaGetDevicePtr(&gpu_id);
    if (res != cudaSuccess) {
        return PAPI_ENOSUPP;
    }
    res = cuCtxGetCurrentPtr(&tempCtx);
    if (res != CUDA_SUCCESS) {
        return PAPI_ENOSUPP;
    }
    if (cu_ctx[gpu_id] == NULL) {
        if (tempCtx != NULL) {
            LOGDBG("Registering device = %d with ctx = %p.\n", gpu_id, tempCtx);
            CUDA_CALL(cuCtxGetCurrentPtr(&cu_ctx[gpu_id]), return PAPI_ENOSUPP);
        }
        else {
            CUDART_CALL(cudaFreePtr(NULL), return PAPI_ENOSUPP);
            CUDA_CALL(cuCtxGetCurrentPtr(&cu_ctx[gpu_id]), return PAPI_ENOSUPP);
            LOGDBG("Using primary device context %p for device %d.\n", cu_ctx[gpu_id], gpu_id);
        }
    }
    else if (cu_ctx[gpu_id] != tempCtx) {  // If context has changed keep the first seen one but with warning
        ERRDBG("Warning: cuda context for gpu %d has changed from %p to %p\n", gpu_id, cu_ctx[gpu_id], tempCtx);
    }

    // Update event names to be profiled for corresponding gpus
    res = add_events_per_gpu(all_event_names, event_count, evt_ids, state);
    if (res != PAPI_OK)
        goto fn_exit;
    // Validate initialized profiler state
    res = control_state_validate(state);

fn_exit:
    *pctl = (void *) state;
    return res;
}

int cuptip_control_destroy(void **pctl)
{
    COMPDBG("Entering.\n");
    cuptip_control_t *state = (cuptip_control_t *) (*pctl);
    int i, j;
    for (i=0; i<num_gpus; i++) {
        reset_cupti_prof_config_images(&(state->ctl[i]));
        free_event_name_list( &(state->ctl[i].event_names) );
        for (j=0; j < state->ctl[i].rmr_count; j++) {
            papi_free((void *) state->ctl[i].rmr[j].pMetricName);
        }
        papi_free(state->ctl[i].rmr);
    }
    papi_free(state->ctl);
    papi_free(state);
    *pctl = NULL;
    return PAPI_OK;
}

int cuptip_start(void **pctl, void **pcu_ctx)
{
    COMPDBG("Entering.\n");
    cuptip_control_t *state = (cuptip_control_t *) (*pctl);
    cuptip_gpu_control_t *ctl;
    CUcontext * cu_ctx = (CUcontext *) (*pcu_ctx);
    CUcontext userCtx;
    CUDA_CALL(cuCtxGetCurrentPtr(&userCtx), return PAPI_ENOSUPP);
    if (userCtx == NULL) {
        CUDART_CALL(cudaFreePtr(NULL), return PAPI_ENOSUPP);
        CUDA_CALL(cuCtxGetCurrentPtr(&userCtx), return PAPI_ENOSUPP);
    }
    int gpu_id;
    char chip_name[32];
    int res = PAPI_OK;
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        ctl = &(state->ctl[gpu_id]);
        LOGDBG("Device num %d: event_count %d, rmr count %d\n", gpu_id, ctl->event_names.count, ctl->rmr_count);
        if (ctl->event_names.count == 0)
            continue;
        res = devmask_check_and_acquire(&(state->ctl[gpu_id].event_names));
        if (res != PAPI_OK) {
            ERRDBG("Profiling same gpu from multiple event sets not allowed.\n");
            return res;
        }
        res = get_chip_name(gpu_id, chip_name);
        if (res != PAPI_OK) {
            ERRDBG("Failed to get chipname.\n");
            goto fn_fail;
        }
        CUDA_CALL(cuCtxSetCurrentPtr(cu_ctx[gpu_id]), return PAPI_ENOSUPP);
        res = get_counter_availability(ctl);
        if (res != PAPI_OK) {
            ERRDBG("Error getting counter availability image.\n");
            return res;
        }
        // CUPTI profiler host configuration
        res = metric_get_config_image(chip_name, ctl);
        res += metric_get_counter_data_prefix_image(chip_name, ctl);
        res += create_counter_data_image(ctl);
        if (res != PAPI_OK) {
            ERRDBG("Failed to create CUPTI profiler state for gpu %d\n", gpu_id);
            goto fn_fail;
        }
        LOGDBG("%d\t%d\t%d\n", ctl->configImage.size, ctl->counterDataScratchBuffer.size,
                               ctl->counterDataImage.size);
        res = begin_profiling(ctl);
        if (res != PAPI_OK) {
            ERRDBG("Failed to start profiling for gpu %d\n", gpu_id);
            goto fn_fail;
        }
    }
    state->running = True;
    CUDA_CALL(cuCtxSetCurrentPtr(userCtx), return PAPI_ENOSUPP);
    return PAPI_OK;
fn_fail:
    CUDA_CALL(cuCtxSetCurrentPtr(userCtx), return PAPI_ENOSUPP);
    return PAPI_ECMP;
}

int cuptip_stop(void **pctl, void **pcu_ctx)
{
    COMPDBG("Entering.\n");
    cuptip_control_t *state = (cuptip_control_t *) (*pctl);
    cuptip_gpu_control_t *ctl;
    CUcontext * cu_ctx = (CUcontext *) (*pcu_ctx);
    CUcontext userCtx;
    CUDA_CALL(cuCtxGetCurrentPtr(&userCtx), return PAPI_ENOSUPP);
    if (userCtx == NULL) {
        CUDART_CALL(cudaFreePtr(NULL), return PAPI_ENOSUPP);
        CUDA_CALL(cuCtxGetCurrentPtr(&userCtx), return PAPI_ENOSUPP);
    }
    int gpu_id;
    int res = PAPI_OK;
    if (state->running == False) {
        ERRDBG("Profiler is already stopped.\n");
        goto fn_fail;
    }
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        ctl = &(state->ctl[gpu_id]);
        if (ctl->event_names.count == 0)
            continue;
        CUDA_CALL(cuCtxSetCurrentPtr(cu_ctx[gpu_id]), return PAPI_ENOSUPP);
        res = end_profiling(ctl);
        if (res != PAPI_OK) {
            ERRDBG("Failed to stop profiling on gpu %d\n", gpu_id);
            goto fn_fail;
        }
        res = devmask_release(&(state->ctl[gpu_id].event_names));
        if (res != PAPI_OK)
            goto fn_fail;
    }
    state->running = False;
    CUDA_CALL(cuCtxSetCurrentPtr(userCtx), return PAPI_ENOSUPP);
    return res;
fn_fail:
    CUDA_CALL(cuCtxSetCurrentPtr(userCtx), return PAPI_ENOSUPP);
    return PAPI_ECMP;
}

static enum collection_method_e get_event_collection_method(const char *evt_name)
{
    if (strstr(evt_name, ".sum") != NULL)
        return RunningSum;
    else if (strstr(evt_name, ".min") != NULL)
        return RunningMin;
    else if (strstr(evt_name, ".max") != NULL)
        return RunningMax;
    else
        return SpotValue;
}

int cuptip_control_read(void **pctl, long long *values)
{
    COMPDBG("Entering.\n");
    int res, gpu_id, i;
    cuptip_control_t *state = (cuptip_control_t *) (*pctl);
    cuptip_gpu_control_t *ctl;
    unsigned int evt_pos;
    long long val;
    res = get_metric_values(state);
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        ctl = &(state->ctl[gpu_id]);
        if (ctl->event_names.count == 0)
            continue;
        for (i=0; i < (int) ctl->event_names.count; i++) {
            evt_pos = ctl->event_names.evts[i].evt_pos;
            val = (long long) ctl->event_names.evts[i].value;

            if (state->read_count == 0) {
                values[evt_pos] = val;
            }
            else {
                switch (get_event_collection_method(ctl->event_names.evts[i].name)) {
                    case RunningSum:
                        values[evt_pos] += val;
                        break;
                    case RunningMin:
                        values[evt_pos] = values[evt_pos] < val ? values[evt_pos] : val;
                        break;
                    case RunningMax:
                        values[evt_pos] = values[evt_pos] > val ? values[evt_pos] : val;
                        break;
                    default:
                        values[evt_pos] = val;
                        break;
                }
            }
        }
        reset_cupti_prof_config_images(ctl);
    }
    state->read_count++;
    return res;
}

int cuptip_control_reset(void **pctl)
{
    COMPDBG("Entering.\n");
    cuptip_control_t *state = (cuptip_control_t *) (*pctl);
    state->read_count = 0;
    return PAPI_OK;
}

void cuptip_shutdown(void)
{
    COMPDBG("Entering.\n");
    free_all_enumerated_metrics();
    deinitialize_cupti_profiler_api();
    unload_nvpw_sym();
    unload_cupti_perf_sym();
}
