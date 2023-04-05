/*
 * Contain only functions related to cuda profiler API
 */

#include <papi.h>
#include "papi_memory.h"
#include "cuda_utils.h"
#include "cupti_profiler.h"
#include "debug_comp.h"
#include <dlfcn.h>

static int load_cupti_perf_sym(void);
static int load_nvpw_sym(void);
static int initialize_cupti_profiler_api(void);
static int initialize_perfworks_api(void);
static int get_chip_name(int dev_num, char* chipName);

static int num_gpus;

NVPA_Status ( *NVPW_GetSupportedChipNamesPtr ) (NVPW_GetSupportedChipNames_Params* params);
NVPA_Status ( *NVPW_CUDA_MetricsContext_CreatePtr ) (NVPW_CUDA_MetricsContext_Create_Params* params);
NVPA_Status ( *NVPW_MetricsContext_DestroyPtr ) (NVPW_MetricsContext_Destroy_Params * params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_BeginPtr ) (NVPW_MetricsContext_GetMetricNames_Begin_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricNames_EndPtr ) (NVPW_MetricsContext_GetMetricNames_End_Params* params);
NVPA_Status ( *NVPW_InitializeHostPtr ) (NVPW_InitializeHost_Params* params);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_BeginPtr ) (NVPW_MetricsContext_GetMetricProperties_Begin_Params* p);
NVPA_Status ( *NVPW_MetricsContext_GetMetricProperties_EndPtr ) (NVPW_MetricsContext_GetMetricProperties_End_Params* p);
NVPA_Status ( *NVPW_CUDA_RawMetricsConfig_CreatePtr ) (NVPW_CUDA_RawMetricsConfig_Create_Params*);

// NVPA_Status ( *NVPA_RawMetricsConfig_CreatePtr ) (NVPA_RawMetricsConfigOptions*, NVPA_RawMetricsConfig**));
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
        LOGCUPTICALL("%s\t" #call "\033[0m\n", __func__);  \
        if (_status != NVPA_STATUS_SUCCESS) {  \
            ANSIRED;  \
            fprintf(stderr, "NVPA Error %d: %s: %d: Error in call to " #call "\n", _status, __FILE__, __LINE__);  \
            ANSIEND;  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

static void *dl4;

static int load_cupti_perf_sym(void)
{
    COMPDBG("Entering.\n");
    int papiErr = PAPI_OK;
    if (dl3 == NULL) {
        ERRDBG("libcupti.so should already be loaded.\n");
        goto fn_fail;
    }

    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl3, "cuptiDeviceGetChipName");
    cuptiProfilerInitializePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerDeInitialize");
    cuptiProfilerCounterDataImageCalculateSizePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerEndSession");
    cuptiProfilerGetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl3, "cuptiProfilerGetCounterAvailability");
    cuptiFinalizePtr = DLSYM_AND_CHECK(dl3, "cuptiFinalize");

fn_exit:
    return papiErr;
fn_fail:
    papiErr = PAPI_ENOSUPP;
    goto fn_exit;
}

static int load_nvpw_sym(void)
{
    COMPDBG("Entering.\n");
    int papiErr = PAPI_OK;
    dl4 = dlopen("libnvperf_host.so", RTLD_NOW | RTLD_GLOBAL);
    if (dl4 == NULL) {
        ERRDBG("Loading libnvperf_host.so failed.\n");
        goto fn_fail;
    }

    NVPW_GetSupportedChipNamesPtr = DLSYM_AND_CHECK(dl4, "NVPW_GetSupportedChipNames");
    NVPW_CUDA_MetricsContext_CreatePtr = DLSYM_AND_CHECK(dl4, "NVPW_CUDA_MetricsContext_Create");
    NVPW_MetricsContext_DestroyPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_InitializeHostPtr = DLSYM_AND_CHECK(dl4, "NVPW_InitializeHost");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_GetMetricProperties_End");
    NVPW_CUDA_RawMetricsConfig_CreatePtr = DLSYM_AND_CHECK(dl4, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetricsPtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_GetConfigImage");
    NVPW_CounterDataBuilder_CreatePtr = DLSYM_AND_CHECK(dl4, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr = DLSYM_AND_CHECK(dl4, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr = DLSYM_AND_CHECK(dl4, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr = DLSYM_AND_CHECK(dl4, "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    NVPW_CounterData_GetNumRangesPtr = DLSYM_AND_CHECK(dl4, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = DLSYM_AND_CHECK(dl4, "NVPW_Profiler_CounterData_GetRangeDescriptions");
    NVPW_MetricsContext_SetCounterDataPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_RawMetricsConfig_GetNumPassesPtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_SetCounterAvailability");
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr = DLSYM_AND_CHECK(dl4, "NVPW_RawMetricsConfig_IsAddMetricsPossible");
    NVPW_MetricsContext_GetCounterNames_BeginPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_GetCounterNames_Begin");
    NVPW_MetricsContext_GetCounterNames_EndPtr = DLSYM_AND_CHECK(dl4, "NVPW_MetricsContext_GetCounterNames_End");

fn_exit:
    return papiErr;
fn_fail:
    papiErr = PAPI_ENOSUPP;
    goto fn_exit;

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

typedef struct byte_array_s {
    uint8_t *data;
    int size;
} byte_array_t;

struct cupti_gpu_control_s {
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

struct cupti_profiler_control_s {
    struct cupti_gpu_control_s *ctl;
    int read_count;
};

static int add_events_per_gpu(struct event_name_list_s * event_names, int event_count, int *evt_ids, struct cupti_profiler_control_s *state)
{
    int i, gpu_id, res = PAPI_OK;
    char nvName[PAPI_MAX_STR_LEN];
    for (i=0; i < num_gpus; i++) {
        res = initialize_dynamic_event_list( &(state->ctl[i].event_names) );
    }
    for (i = 0; i < event_count; i++) {
        res = tokenize_event_name(event_names->evts[evt_ids[i]].name, (char*) &nvName, &gpu_id);
        if (res != PAPI_OK)
            goto fn_exit;
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

static int get_event_names_rmr(struct NVPA_MetricsContext* pMetricsContext, struct cupti_gpu_control_s *ctl)
{
    int res = PAPI_OK;

    struct cupti_gpu_control_s *rmr_per_event;  // Array of dependent raw metrics for each event name
    int count_raw_metrics = 0;
    unsigned int i;
    int j, num_dep;
    NVPA_RawMetricRequest *temp;
    char nv_name[PAPI_MAX_STR_LEN]; int gpuid;

    rmr_per_event = (struct cupti_gpu_control_s *) papi_calloc(ctl->event_names.count, sizeof(struct cupti_gpu_control_s));
    if (rmr_per_event == NULL) {
        res = PAPI_ENOMEM;
        goto fn_exit;
    }
    for (i=0; i < ctl->event_names.count; i++) {
        res = tokenize_event_name(ctl->event_names.evts[i].name, (char *) &nv_name, &gpuid);

        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = {
            .structSize = NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = pMetricsContext,
            .pMetricName = nv_name,
        };
        res = NVPW_MetricsContext_GetMetricProperties_BeginPtr(&getMetricPropertiesBeginParams);

        // Check output of GetMetricProperties_Begin after calling End
        if (res != NVPA_STATUS_SUCCESS ||
            getMetricPropertiesBeginParams.ppRawMetricDependencies == NULL) {
                ERRDBG("Please check event name %s\n", ctl->event_names.evts[i].name);
                res = PAPI_EINVAL;
                goto fn_exit;
        }

        // Count number of dependent raw metrics for this event
        for (num_dep=0;
            getMetricPropertiesBeginParams.ppRawMetricDependencies[num_dep] != NULL;
            num_dep++);
        rmr_per_event[i].rmr_count = num_dep;
        count_raw_metrics += num_dep;

        rmr_per_event[i].rmr = (NVPA_RawMetricRequest *) papi_calloc(num_dep, sizeof(NVPA_RawMetricRequest));
        if (rmr_per_event[i].rmr == NULL) {
            res = PAPI_ENOMEM;
            goto fn_exit;
        }

        for (j=0; j<num_dep; j++) {
            temp = &( rmr_per_event[i].rmr[j] );
            temp->pMetricName = strdup(getMetricPropertiesBeginParams.ppRawMetricDependencies[j]);
            temp->isolated = 1;
            temp->keepInstances = 1;
            temp->structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE;
        }
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = {
            .structSize = NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .pMetricsContext = pMetricsContext,
        };
        NVPW_CALL(NVPW_MetricsContext_GetMetricProperties_EndPtr(&getMetricPropertiesEndParams),
            return PAPI_ENOSUPP);

    }
    // Collect and build array of all raw metric requests
    NVPA_RawMetricRequest * all_rmr = (NVPA_RawMetricRequest *) papi_calloc(count_raw_metrics, sizeof(NVPA_RawMetricRequest));
    if (all_rmr == NULL) {
        res = PAPI_ENOMEM;
        goto fn_exit;
    }
    int count = 0;
    for (i=0; i < ctl->event_names.count; i++) {
        for (j=0; j < rmr_per_event[i].rmr_count; j++) {
            temp = &( rmr_per_event[i].rmr[j] );
            all_rmr[count].structSize = temp->structSize;
            all_rmr[count].pMetricName = strdup(temp->pMetricName);
            all_rmr[count].keepInstances = 1;
            all_rmr[count].isolated = 1;
            count++;
            papi_free((void *) temp->pMetricName);
        }
    }
    // Free the consumed memory
    for (i=0; i<ctl->event_names.count; i++) {
        papi_free(rmr_per_event[i].rmr);
    }
    papi_free(rmr_per_event);
    ctl->rmr = all_rmr;
    ctl->rmr_count = count;
fn_exit:
    return res;
}

static int check_num_passes(struct NVPA_RawMetricsConfig* pRawMetricsConfig, struct cupti_gpu_control_s *ctl)
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
        .pRawMetricRequests = ctl->rmr,
        .numMetricRequests = ctl->rmr_count,
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

    int numNestingLevels = 1, numIsolatedPasses, numPipelinedPasses, numOfPasses;
    numIsolatedPasses  = rawMetricsConfigGetNumPassesParams.numIsolatedPasses;
    numPipelinedPasses = rawMetricsConfigGetNumPassesParams.numPipelinedPasses;

    numOfPasses = numPipelinedPasses + numIsolatedPasses * numNestingLevels;

    if (numOfPasses > 1) {
        ERRDBG("error: Metrics requested requires multiple passes to profile.\n");
        return PAPI_EMULPASS;
    }

    return PAPI_OK;
}

static int control_state_validate(struct cupti_profiler_control_s *state)
{
    COMPDBG("Entering.\n");
    int gpu_id, res = PAPI_OK;
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

        NVPW_CUDA_RawMetricsConfig_Create_Params nvpw_metricsConfigCreateParams = {
            .structSize = NVPW_CUDA_RawMetricsConfig_Create_Params_STRUCT_SIZE,
            .pPriv = NULL,
            .activityKind = NVPA_ACTIVITY_KIND_PROFILER,
            .pChipName = chip_name,
        };
        NVPW_CALL( NVPW_CUDA_RawMetricsConfig_CreatePtr(&nvpw_metricsConfigCreateParams), return PAPI_ENOSUPP );

        res = check_num_passes(nvpw_metricsConfigCreateParams.pRawMetricsConfig, &(state->ctl[gpu_id]));

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

// Adapted from `metric.cpp` -------------------------
static int metric_get_config_image(const char *chipName, struct cupti_gpu_control_s *ctl)
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

static int metric_get_counter_data_prefix_image(const char* chipName, struct cupti_gpu_control_s *ctl)
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
static int create_counter_data_image(struct cupti_gpu_control_s *ctl)
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

static int reset_cupti_prof_config_images(struct cupti_gpu_control_s *ctl)
{
    papi_free(ctl->counterDataImagePrefix.data);
    papi_free(ctl->configImage.data);
    papi_free(ctl->counterDataImage.data);
    papi_free(ctl->counterDataScratchBuffer.data);
    papi_free(ctl->counterAvailabilityImage.data);
    return PAPI_OK;
}

static int begin_profiling(struct cupti_gpu_control_s *ctl, CUcontext cuctx)
{
    byte_array_t *configImage = &(ctl->configImage);
    byte_array_t *counterDataScratchBuffer = &(ctl->counterDataScratchBuffer);
    byte_array_t *counterDataImage = &(ctl->counterDataImage);

    CUDA_CALL( cuCtxSetCurrentPtr(cuctx), return PAPI_ECMP );
    LOGDBG("dev = %d has current ctx = %p.\n", ctl->gpu_id, cuctx);

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

static int end_profiling(struct cupti_gpu_control_s *ctl, CUcontext cuctx)
{

    COMPDBG("EndProfiling. dev = %d ctx = %p\n", ctl->gpu_id, cuctx);
    (void) ctl;
    CUDA_CALL( cuCtxSetCurrentPtr(cuctx), return PAPI_ECMP );

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
static int eval_metric_values_per_gpu(struct NVPA_MetricsContext* pMetricsContext, struct cupti_gpu_control_s *ctl)
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

static int get_metric_values(struct cupti_profiler_control_s *state)
{
    COMPDBG("Entering.\n");
    int gpu_id, res = PAPI_OK, i;
    char chip_name[32];
    struct cupti_gpu_control_s *ctl;
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

// CUPTI Profiler component API functions
int cupti_profiler_init(const char ** pdisabled_reason)
{
    int retval = PAPI_OK;
    retval = get_env_papi_cuda_root();
    if (retval != PAPI_OK) {
        *pdisabled_reason = "Environment variable PAPI_CUDA_ROOT not set.";
        goto fn_fail;
    }
    retval = load_cuda_sym();
    retval += load_cudart_sym();
    retval += load_cupti_common_sym();
    retval += load_cupti_perf_sym();
    retval += load_nvpw_sym();
    if (retval != PAPI_OK) {
        *pdisabled_reason = "Unable to load CUDA library functions.";
        goto fn_fail;
    }
    retval = check_cuda_api_versions();
    if (retval == PAPI_ECMP) {
        *pdisabled_reason = "CUDA library included vs linked versions mismatch.";
        goto fn_fail;
    }
    else if (retval == PAPI_ENOSUPP) {
        *pdisabled_reason = "CUDA driver version older than runtime version.";
        goto fn_fail;
    }
    retval = get_device_count();
    if (retval <= 0) {
        *pdisabled_reason = "No GPUs found on system.";
        goto fn_fail;
    }
    num_gpus = retval;
    if (is_mixed_compute_capability() != PAPI_OK) {
        *pdisabled_reason = "Mixed compute capability not supported.";
        goto fn_fail;
    }
    retval = initialize_cupti_profiler_api();
    retval += initialize_perfworks_api();
    if (retval != PAPI_OK) {
        *pdisabled_reason = "Unable to initialize CUPTI profiler libraries.";
        goto fn_fail;
    }
    return PAPI_OK;
fn_fail:
    return PAPI_ECMP;
}

int cupti_profiler_control_create(struct event_name_list_s * all_event_names, int event_count, int *evt_ids, void ** pctl, void **pcu_ctx)
{
    COMPDBG("Entering.\n");
    int res = PAPI_OK, gpu_id;
    // Allocate profiler control state
    struct cupti_profiler_control_s * state = (struct cupti_profiler_control_s *) papi_calloc (1, sizeof(struct cupti_profiler_control_s));
    if (state == NULL) {
        return PAPI_ENOMEM;
    }
    // Allocate array of profiler states for each gpu
    state->ctl = (struct cupti_gpu_control_s *) papi_calloc(num_gpus, sizeof(struct cupti_gpu_control_s));
    if (state->ctl == NULL) {
        return PAPI_ENOMEM;
    }
    // Set gpu_id for each profiler state
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++)
        state->ctl[gpu_id].gpu_id = gpu_id;

    // Register the user created cuda context for the current gpu if not already known
    CUcontext *cu_ctx = (CUcontext *) (*pcu_ctx);
    CUcontext temp;
    res = cudaGetDevicePtr(&gpu_id);
    res = cuCtxGetCurrentPtr(&temp);
    if (cu_ctx[gpu_id] == NULL) {
        if (temp != NULL) {
            LOGDBG("Registering device = %d with ctx = %p.\n", gpu_id, temp);
            res = cuCtxGetCurrentPtr(&cu_ctx[gpu_id]);
        }
        else {
            ERRDBG("Got NULL current cuda context for device %d.\n", gpu_id);
        }
    }
    else if (cu_ctx[gpu_id] != temp) {  // If context has changed keep the first seen one but with warning
        ERRDBG("Warning: cuda context for gpu %d has changed from %p to %p\n", gpu_id, cu_ctx[gpu_id], temp);
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

int cupti_profiler_control_destroy(void **pctl)
{
    COMPDBG("Entering.\n");
    struct cupti_profiler_control_s * state = (struct cupti_profiler_control_s *) (*pctl);
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

int cupti_profiler_start(void **pctl, void **pcu_ctx)
{
    COMPDBG("Entering.\n");
    struct cupti_profiler_control_s * state = (struct cupti_profiler_control_s *) (*pctl);
    struct cupti_gpu_control_s *ctl;
    CUcontext * cu_ctx = (CUcontext *) (*pcu_ctx);
    int gpu_id;
    char chip_name[32];
    int res = PAPI_OK;
    // int userDevice;
    // res = cudaGetDevicePtr(&userDevice);
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++)
    {
        ctl = &(state->ctl[gpu_id]);
        LOGDBG("Device num %d: event_count %d, rmr count %d\n", gpu_id, ctl->event_names.count, ctl->rmr_count);
        if (ctl->event_names.count == 0)
            continue;
        res = get_chip_name(gpu_id, chip_name);
        if (res != PAPI_OK) {
            ERRDBG("Failed to get chipname.\n");
            goto fn_fail;
        }
        // CUPTI profiler host configuration
        res = metric_get_config_image(chip_name, ctl);
        res = metric_get_counter_data_prefix_image(chip_name, ctl);
        res = create_counter_data_image(ctl);
        LOGDBG("%d\t%d\t%d\n", ctl->configImage.size, ctl->counterDataScratchBuffer.size,
                               ctl->counterDataImage.size);
        res = begin_profiling(ctl, cu_ctx[gpu_id]);
    }
    // res = cudaSetDevicePtr(userDevice);
    return PAPI_OK;
fn_fail:
    return PAPI_ECMP;
}

int cupti_profiler_stop(void **pctl, void **pcu_ctx)
{
    COMPDBG("Entering.\n");
    struct cupti_profiler_control_s * state = (struct cupti_profiler_control_s *) (*pctl);
    struct cupti_gpu_control_s *ctl;
    CUcontext * cu_ctx = (CUcontext *) (*pcu_ctx);
    int gpu_id;
    int res = PAPI_OK;
    // int userDevice;
    // res = cudaGetDevicePtr(&userDevice);
    for (gpu_id=0; gpu_id<num_gpus; gpu_id++) {
        ctl = &(state->ctl[gpu_id]);
        if (ctl->event_names.count == 0)
            continue;
        res = end_profiling(ctl, cu_ctx[gpu_id]);
    }
    // res = cudaSetDevicePtr(userDevice);
    return res;
}

enum {SpotValue, RunningMin, RunningMax, RunningSum};

static int get_event_collection_method(const char * evt_name)
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

int cupti_profiler_control_read(void **pctl, long long *values)
{
    COMPDBG("Entering.\n");
    int res, gpu_id, i;
    struct cupti_profiler_control_s * state = (struct cupti_profiler_control_s *) (*pctl);
    struct cupti_gpu_control_s *ctl;
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