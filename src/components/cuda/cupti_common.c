/**
 * @file    cupti_common.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <dlfcn.h>
#include <link.h>
#include <libgen.h>
#include <papi.h>
#include "papi_memory.h"

#include "cupti_config.h"
#include "cupti_common.h"

static void *dl_drv, *dl_rt;

const char *linked_cudart_path;
void *dl_cupti;

unsigned int _cuda_lock;

CUresult ( *cuCtxGetCurrentPtr ) (CUcontext *);
CUresult ( *cuCtxSetCurrentPtr ) (CUcontext);
CUresult ( *cuCtxDestroyPtr ) (CUcontext);
CUresult ( *cuCtxCreatePtr ) (CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult ( *cuCtxGetDevicePtr ) (CUdevice *);
CUresult ( *cuDeviceGetPtr ) (CUdevice *, int);
CUresult ( *cuDeviceGetCountPtr ) (int *);
CUresult ( *cuDeviceGetNamePtr ) (char *, int, CUdevice);
CUresult ( *cuDevicePrimaryCtxRetainPtr ) (CUcontext *pctx, CUdevice);
CUresult ( *cuDevicePrimaryCtxReleasePtr ) (CUdevice);
CUresult ( *cuInitPtr ) (unsigned int);
CUresult ( *cuGetErrorStringPtr ) (CUresult error, const char** pStr);
CUresult ( *cuCtxPopCurrentPtr ) (CUcontext * pctx);
CUresult ( *cuCtxPushCurrentPtr ) (CUcontext pctx);
CUresult ( *cuCtxSynchronizePtr ) ();
CUresult ( *cuDeviceGetAttributePtr ) (int *, CUdevice_attribute, CUdevice);

cudaError_t ( *cudaGetDeviceCountPtr ) (int *);
cudaError_t ( *cudaGetDevicePtr ) (int *);
cudaError_t ( *cudaSetDevicePtr ) (int);
cudaError_t ( *cudaGetDevicePropertiesPtr ) (struct cudaDeviceProp* prop, int  device);
cudaError_t ( *cudaDeviceGetAttributePtr ) (int *value, enum cudaDeviceAttr attr, int device);
cudaError_t ( *cudaFreePtr ) (void *);
cudaError_t ( *cudaDriverGetVersionPtr ) (int *);
cudaError_t ( *cudaRuntimeGetVersionPtr ) (int *);

CUptiResult ( *cuptiGetVersionPtr ) (uint32_t* );

static int load_cuda_sym(void)
{
    dl_drv = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_drv) {
        ERRDBG("Loading installed libcuda.so failed. Check that cuda drivers are installed.\n");
        goto fn_fail;
    }

    cuCtxSetCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxSetCurrent");
    cuCtxGetCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxGetCurrent");
    cuCtxDestroyPtr              = DLSYM_AND_CHECK(dl_drv, "cuCtxDestroy");
    cuCtxCreatePtr               = DLSYM_AND_CHECK(dl_drv, "cuCtxCreate");
    cuCtxGetDevicePtr            = DLSYM_AND_CHECK(dl_drv, "cuCtxGetDevice");
    cuDeviceGetPtr               = DLSYM_AND_CHECK(dl_drv, "cuDeviceGet");
    cuDeviceGetCountPtr          = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetCount");
    cuDeviceGetNamePtr           = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetName");
    cuDevicePrimaryCtxRetainPtr  = DLSYM_AND_CHECK(dl_drv, "cuDevicePrimaryCtxRetain");
    cuDevicePrimaryCtxReleasePtr = DLSYM_AND_CHECK(dl_drv, "cuDevicePrimaryCtxRelease");
    cuInitPtr                    = DLSYM_AND_CHECK(dl_drv, "cuInit");
    cuGetErrorStringPtr          = DLSYM_AND_CHECK(dl_drv, "cuGetErrorString");
    cuCtxPopCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr          = DLSYM_AND_CHECK(dl_drv, "cuCtxPushCurrent");
    cuCtxSynchronizePtr          = DLSYM_AND_CHECK(dl_drv, "cuCtxSynchronize");
    cuDeviceGetAttributePtr      = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetAttribute");

    Dl_info info;
    dladdr(cuCtxSetCurrentPtr, &info);
    LOGDBG("CUDA driver library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cuda_sym(void)
{
    if (dl_drv) {
        dlclose(dl_drv);
        dl_drv = NULL;
    }
    cuCtxSetCurrentPtr           = NULL;
    cuCtxGetCurrentPtr           = NULL;
    cuCtxDestroyPtr              = NULL;
    cuCtxCreatePtr               = NULL;
    cuCtxGetDevicePtr            = NULL;
    cuDeviceGetPtr               = NULL;
    cuDeviceGetCountPtr          = NULL;
    cuDeviceGetNamePtr           = NULL;
    cuDevicePrimaryCtxRetainPtr  = NULL;
    cuDevicePrimaryCtxReleasePtr = NULL;
    cuInitPtr                    = NULL;
    cuGetErrorStringPtr          = NULL;
    cuCtxPopCurrentPtr           = NULL;
    cuCtxPushCurrentPtr          = NULL;
    cuCtxSynchronizePtr          = NULL;
    cuDeviceGetAttributePtr      = NULL;
    return PAPI_OK;
}

void *cuptic_load_dynamic_syms(const char *parent_path, const char *dlname, const char *search_subpaths[])
{
    void *dl = NULL;
    char lookup_path[PATH_MAX];
    char *found_files[CUPTIU_MAX_FILES];
    int i, count;
    for (i = 0; search_subpaths[i] != NULL; i++) {
        sprintf(lookup_path, search_subpaths[i], parent_path, dlname);
        dl = dlopen(lookup_path, RTLD_NOW | RTLD_GLOBAL);
        if (dl) {
            return dl;
        }
    }
    count = cuptiu_files_search_in_path(dlname, parent_path, found_files);
    for (i = 0; i < count; i++) {
        dl = dlopen(found_files[i], RTLD_NOW | RTLD_GLOBAL);
        if (dl) {
            break;
        }
    }
    for (i = 0; i < count; i++) {
        papi_free(found_files[i]);
    }
    return dl;
}

static int load_cudart_sym(void)
{
    char dlname[] = "libcudart.so";
    char lookup_path[PATH_MAX];

    char *papi_cuda_runtime = getenv("PAPI_CUDA_RUNTIME");
    if (papi_cuda_runtime) {
        sprintf(lookup_path, "%s/%s", papi_cuda_runtime, dlname);
        dl_rt = dlopen(lookup_path, RTLD_NOW | RTLD_GLOBAL);
    }

    const char *standard_paths[] = {
        "%s/lib64/%s",
        NULL,
    };

    if (linked_cudart_path && !dl_rt) {
        dl_rt = cuptic_load_dynamic_syms(linked_cudart_path, dlname, standard_paths);
    }

    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_rt) {
        dl_rt = cuptic_load_dynamic_syms(papi_cuda_root, dlname, standard_paths);
    }

    if (!dl_rt) {
        dl_rt = dlopen(dlname, RTLD_NOW | RTLD_GLOBAL);
        if (!dl_rt) {
            ERRDBG("Loading libcudart.so failed. Try setting PAPI_CUDA_ROOT\n");
            goto fn_fail;
        }
    }

    cudaGetDevicePtr           = DLSYM_AND_CHECK(dl_rt, "cudaGetDevice");
    cudaGetDeviceCountPtr      = DLSYM_AND_CHECK(dl_rt, "cudaGetDeviceCount");
    cudaGetDevicePropertiesPtr = DLSYM_AND_CHECK(dl_rt, "cudaGetDeviceProperties");
    cudaDeviceGetAttributePtr  = DLSYM_AND_CHECK(dl_rt, "cudaDeviceGetAttribute");
    cudaSetDevicePtr           = DLSYM_AND_CHECK(dl_rt, "cudaSetDevice");
    cudaFreePtr                = DLSYM_AND_CHECK(dl_rt, "cudaFree");
    cudaDriverGetVersionPtr    = DLSYM_AND_CHECK(dl_rt, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr   = DLSYM_AND_CHECK(dl_rt, "cudaRuntimeGetVersion");

    Dl_info info;
    dladdr(cudaGetDevicePtr, &info);
    LOGDBG("CUDA runtime library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cudart_sym(void)
{
    if (dl_rt) {
        dlclose(dl_rt);
        dl_rt = NULL;
    }
    cudaGetDevicePtr           = NULL;
    cudaGetDeviceCountPtr      = NULL;
    cudaGetDevicePropertiesPtr = NULL;
    cudaDeviceGetAttributePtr  = NULL;
    cudaSetDevicePtr           = NULL;
    cudaFreePtr                = NULL;
    cudaDriverGetVersionPtr    = NULL;
    cudaRuntimeGetVersionPtr   = NULL;
    return PAPI_OK;
}

static int load_cupti_common_sym(void)
{
    char dlname[] = "libcupti.so";
    char lookup_path[PATH_MAX];

    char *papi_cuda_cupti = getenv("PAPI_CUDA_CUPTI");
    if (papi_cuda_cupti) {
        sprintf(lookup_path, "%s/%s", papi_cuda_cupti, dlname);
        dl_cupti = dlopen(lookup_path, RTLD_NOW | RTLD_GLOBAL);
    }

    const char *standard_paths[] = {
        "%s/extras/CUPTI/lib64/%s",
        "%s/lib64/%s",
        NULL,
    };

    if (linked_cudart_path && !dl_cupti) {
        dl_cupti = cuptic_load_dynamic_syms(linked_cudart_path, dlname, standard_paths);
    }

    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_cupti) {
        dl_cupti = cuptic_load_dynamic_syms(papi_cuda_root, dlname, standard_paths);
    }

    if (!dl_cupti) {
        dl_cupti = dlopen(dlname, RTLD_NOW | RTLD_GLOBAL);
        if (!dl_cupti) {
            ERRDBG("Loading libcupti.so failed. Try setting PAPI_CUDA_ROOT\n");
            goto fn_fail;
        }
    }

    cuptiGetVersionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiGetVersion");

    Dl_info info;
    dladdr(cuptiGetVersionPtr, &info);
    LOGDBG("CUPTI library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cupti_common_sym(void)
{
    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
    cuptiGetVersionPtr = NULL;
    return PAPI_OK;
}

static int util_load_cuda_sym(void)
{
    int papi_errno;
    papi_errno = load_cuda_sym();
    papi_errno += load_cudart_sym();
    papi_errno += load_cupti_common_sym();
    if (papi_errno != PAPI_OK) {
        return PAPI_EMISC;
    }
    else
        return PAPI_OK;
}

static void unload_linked_cudart_path(void)
{
    if (linked_cudart_path) {
        papi_free((void*) linked_cudart_path);
        linked_cudart_path = NULL;
    }
}

int cuptic_shutdown(void)
{
    unload_cuda_sym();
    unload_cudart_sym();
    unload_cupti_common_sym();
    unload_linked_cudart_path();
    return PAPI_OK;
}

static int util_dylib_cu_runtime_version(void)
{
    int runtimeVersion;
    CUDART_CALL(cudaRuntimeGetVersionPtr(&runtimeVersion), return PAPI_EMISC );
    return runtimeVersion;
}

static int util_dylib_cupti_version(void)
{
    unsigned int cuptiVersion;
    CUPTI_CALL(cuptiGetVersionPtr(&cuptiVersion), return PAPI_EMISC );
    return cuptiVersion;
}

int cuptic_device_get_count(void)
{
    static int numDevs = -1;
    if (numDevs != -1) {
        goto fn_exit;
    }
    CUDART_CALL(cudaGetDeviceCountPtr(&numDevs), return PAPI_EMISC);
fn_exit:
    return numDevs;
}

static int get_gpu_compute_capability(int dev_num)
{
    int cc_major, cc_minor;
    int cc;
    CUDART_CALL(cudaDeviceGetAttributePtr(&cc_major,
        cudaDevAttrComputeCapabilityMajor, dev_num),
            return PAPI_EMISC );
    CUDART_CALL(cudaDeviceGetAttributePtr(&cc_minor,
        cudaDevAttrComputeCapabilityMinor, dev_num),
            return PAPI_EMISC );
    cc = cc_major * 10 + cc_minor;
    return cc;
}

typedef enum {GPU_COLLECTION_UNKNOWN, GPU_COLLECTION_ALL_PERF, GPU_COLLECTION_MIXED, GPU_COLLECTION_ALL_EVENTS, GPU_COLLECTION_ALL_CC70} gpu_collection_e;

static gpu_collection_e util_gpu_collection_kind(void)
{
    static gpu_collection_e kind = GPU_COLLECTION_UNKNOWN;
    if (kind != GPU_COLLECTION_UNKNOWN) {
        goto fn_exit;
    }

    int total_gpus = cuptic_device_get_count();
    int i, cc;
    int count_perf = 0, count_evt = 0, count_cc70 = 0;
    for (i=0; i<total_gpus; i++) {
        cc = get_gpu_compute_capability(i);
        if (cc == 70) {
            ++count_cc70;
        }
        if (cc >= 70) {
            ++count_perf;
        }
        if (cc <= 70) {
            ++count_evt;
        }
    }
    if (count_cc70 == total_gpus) {
        kind = GPU_COLLECTION_ALL_CC70;
        goto fn_exit;
    }
    if (count_perf == total_gpus) {
        kind = GPU_COLLECTION_ALL_PERF;
        goto fn_exit;
    }
    if (count_evt == total_gpus) {
        kind = GPU_COLLECTION_ALL_EVENTS;
        goto fn_exit;
    }
    kind = GPU_COLLECTION_MIXED;

fn_exit:
    return kind;
}

const char *cuptic_disabled_reason_g;

void cuptic_disabled_reason_set(const char *msg)
{
    cuptic_disabled_reason_g = msg;
}

void cuptic_disabled_reason_get(const char **pmsg)
{
    *pmsg = cuptic_disabled_reason_g;
}

static int dl_iterate_phdr_cb(struct dl_phdr_info *info, __attribute__((unused)) size_t size, __attribute__((unused)) void *data)
{
    const char *library_name = "libcudart.so";
    const char *library_path = info->dlpi_name;

    if (library_path != NULL && strstr(library_path, library_name) != NULL) {
        linked_cudart_path = strdup(dirname(dirname((char *) library_path)));
    }

    return PAPI_OK;
}

static int get_user_cudart_path(void)
{
    dl_iterate_phdr(dl_iterate_phdr_cb, NULL);
    if (NULL == linked_cudart_path) {
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int cuptic_init(void)
{
    int papi_errno = get_user_cudart_path();
    if (papi_errno == PAPI_OK) {
        LOGDBG("Linked cudart root: %s\n", linked_cudart_path);
    }
    else {
        LOGDBG("Target application not linked with cuda runtime libraries.\n");
    }
    papi_errno = util_load_cuda_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_disabled_reason_set("Unable to load CUDA library functions.");
        goto fn_exit;
    }

    if (util_gpu_collection_kind() == GPU_COLLECTION_MIXED) {
        cuptic_disabled_reason_set("No support for systems with mixed compute capabilities, such as CC < 7.0 and CC > 7.0 GPUS.");
        papi_errno = PAPI_ECMP;
        goto fn_exit;
    }
fn_exit:
    return papi_errno;
}

int cuptic_is_runtime_perfworks_api(void)
{
    static int is_perfworks_api = -1;
    if (is_perfworks_api != -1) {
        goto fn_exit;
    }
    char *papi_cuda_110_cc70_perfworks_api = getenv("PAPI_CUDA_110_CC_70_PERFWORKS_API");

    gpu_collection_e gpus_kind = util_gpu_collection_kind();
    unsigned int cuptiVersion = util_dylib_cupti_version();

    if (gpus_kind == GPU_COLLECTION_ALL_CC70 && 
        (cuptiVersion == CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION || util_dylib_cu_runtime_version() == 11000))
    {
        if (papi_cuda_110_cc70_perfworks_api != NULL) {
            is_perfworks_api = 1;
            goto fn_exit;
        }
        else {
            is_perfworks_api = 0;
            goto fn_exit;
        }
    }

    if ((gpus_kind == GPU_COLLECTION_ALL_PERF || gpus_kind == GPU_COLLECTION_ALL_CC70) && cuptiVersion >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION) {
        is_perfworks_api = 1;
        goto fn_exit;
    } else {
        is_perfworks_api = 0;
        goto fn_exit;
    }

fn_exit:
    return is_perfworks_api;
}

int cuptic_is_runtime_events_api(void)
{
    static int is_events_api = -1;
    if (is_events_api != -1) {
        goto fn_exit;
    }

    gpu_collection_e gpus_kind = util_gpu_collection_kind();

    /*
     * See cupti_config.h: When NVIDIA removes the events API add a check in the following condition
     * to check the `util_dylib_cupti_version()` is also <= CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION.
     */
    if ((gpus_kind == GPU_COLLECTION_ALL_EVENTS || gpus_kind == GPU_COLLECTION_ALL_CC70)) {
        is_events_api = 1;
        goto fn_exit;
    } else {
        is_events_api = 0;
        goto fn_exit;
    }
fn_exit:
    return is_events_api;
}

struct cuptic_info {
    CUcontext ctx;
};

int cuptic_ctxarr_create(cuptic_info_t *pinfo)
{
    COMPDBG("Entering.\n");
    cuptic_info_t cuCtx = (cuptic_info_t) papi_calloc (cuptic_device_get_count(), sizeof(*pinfo));
    if (cuCtx == NULL) {
        return PAPI_ENOMEM;
    }
    *pinfo = cuCtx;
    return PAPI_OK;
}

int cuptic_ctxarr_update_current(cuptic_info_t info)
{
    int papi_errno, gpu_id;
    CUcontext tempCtx;
    papi_errno = cudaGetDevicePtr(&gpu_id);
    if (papi_errno != cudaSuccess) {
        return PAPI_EMISC;
    }
    papi_errno = cuCtxGetCurrentPtr(&tempCtx);
    if (papi_errno != CUDA_SUCCESS) {
        return PAPI_EMISC;
    }
    if (info[gpu_id].ctx == NULL) {
        if (tempCtx != NULL) {
            LOGDBG("Registering device = %d with ctx = %p.\n", gpu_id, tempCtx);
            CUDA_CALL(cuCtxGetCurrentPtr(&info[gpu_id].ctx), return PAPI_EMISC);
        }
        else {
            CUDART_CALL(cudaFreePtr(NULL), return PAPI_EMISC);
            CUDA_CALL(cuCtxGetCurrentPtr(&info[gpu_id].ctx), return PAPI_EMISC);
            LOGDBG("Using primary device context %p for device %d.\n", info[gpu_id].ctx, gpu_id);
        }
    }
    /* If context has changed keep the first seen one but with warning */
    else if (info[gpu_id].ctx != tempCtx) {
        ERRDBG("Warning: cuda context for gpu %d has changed from %p to %p\n", gpu_id, info[gpu_id].ctx, tempCtx);
    }
    return PAPI_OK;
}

int cuptic_ctxarr_get_ctx(cuptic_info_t info, int gpu_idx, CUcontext *ctx)
{
    *ctx = info[gpu_idx].ctx;
    return PAPI_OK;
}

int cuptic_ctxarr_destroy(cuptic_info_t *pinfo)
{
    papi_free(*pinfo);
    *pinfo = NULL;
    return PAPI_OK;
}

/* Functions based on bitmasking to detect gpu exclusivity */
typedef int64_t gpu_occupancy_t;
static gpu_occupancy_t global_gpu_bitmask;

static int event_name_get_gpuid(const char *name, int *gpuid)
{
    int papi_errno = PAPI_OK;
    char *token;
    char *copy = strdup(name);

    token = strtok(copy, "=");
    if (token == NULL) {
        goto fn_fail;
    }
    token = strtok(NULL, "\0");
    if (token == NULL) {
        goto fn_fail;
    }
    *gpuid = strtol(token, NULL, 10);

fn_exit:
    papi_free(copy);
    return papi_errno;
fn_fail:
    papi_errno = PAPI_EINVAL;
    goto fn_exit;
}

static int _devmask_events_get(cuptiu_event_table_t *evt_table, gpu_occupancy_t *bitmask)
{
    int papi_errno = PAPI_OK, gpu_id;
    long i;
    gpu_occupancy_t acq_mask = 0;
    cuptiu_event_t *evt_rec;
    for (i = 0; i < evt_table->count; i++) {
        papi_errno = cuptiu_event_table_get_item(evt_table, i, (cuptiu_event_t **) &evt_rec);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        papi_errno = event_name_get_gpuid(evt_rec->name, &gpu_id);
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }
        acq_mask |= (1 << gpu_id);
    }
    *bitmask = acq_mask;
fn_exit:
    return papi_errno;
}

int cuptic_device_acquire(cuptiu_event_table_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int papi_errno = _devmask_events_get(evt_table, &bitmask);
    if (papi_errno != PAPI_OK)
        return papi_errno;
    if (bitmask & global_gpu_bitmask) {
        return PAPI_ECNFLCT;
    }
    _papi_hwi_lock(_cuda_lock);
    global_gpu_bitmask |= bitmask;
    _papi_hwi_unlock(_cuda_lock);
    return PAPI_OK;
}

int cuptic_device_release(cuptiu_event_table_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int papi_errno = _devmask_events_get(evt_table, &bitmask);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    if ((bitmask & global_gpu_bitmask) != bitmask) {
        return PAPI_EMISC;
    }
    _papi_hwi_lock(_cuda_lock);
    global_gpu_bitmask ^= bitmask;
    _papi_hwi_unlock(_cuda_lock);
    return PAPI_OK;
}
