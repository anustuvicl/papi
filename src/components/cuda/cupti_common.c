/**
 * @file    cupti_common.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <dlfcn.h>
#include <papi.h>
#include "papi_memory.h"

#include "lcuda_config.h"
#include "cupti_common.h"
#include "lcuda_common.h"

static void *dl_drv, *dl_rt;

void *dl_cupti;

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

void* load_dynamic_syms(const char *parent_path, const char *dlname, const char *search_subpaths[])
{
    void *dl = NULL;
    char lookup_path[PATH_MAX];
    char *found_files[MAX_FILES];
    int i, count;
    for (i = 0; search_subpaths[i] != NULL; i++) {
        sprintf(lookup_path, search_subpaths[i], parent_path, dlname);
        dl = dlopen(lookup_path, RTLD_NOW | RTLD_GLOBAL);
        if (dl) {
            return dl;
        }
    }
    count = search_files_in_path(dlname, parent_path, found_files);
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

    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_rt) {
        dl_rt = load_dynamic_syms(papi_cuda_root, dlname, standard_paths);
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

    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_cupti) {
        dl_cupti = load_dynamic_syms(papi_cuda_root, dlname, standard_paths);
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

int util_unload_cuda_sym(void)
{
    unload_cuda_sym();
    unload_cudart_sym();
    unload_cupti_common_sym();
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

int get_device_count(void)
{
    static int numDevs = -1;
    if (numDevs != -1)
        goto fn_exit;
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

enum gpu_collection_e {GPU_COLLECTION_UNKNOWN, GPU_COLLECTION_ALL_PERF, GPU_COLLECTION_MIXED, GPU_COLLECTION_ALL_EVENTS, GPU_COLLECTION_ALL_CC70};

static enum gpu_collection_e util_gpu_collection_kind(void)
{
    static enum gpu_collection_e kind = GPU_COLLECTION_UNKNOWN;
    if (kind != GPU_COLLECTION_UNKNOWN)
        goto fn_exit;

    int total_gpus = get_device_count();
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

int cupti_common_init(const char **pdisabled_reason)
{
    int papi_errno = util_load_cuda_sym();
    if (papi_errno != PAPI_OK) {
        *pdisabled_reason = "Unable to load CUDA library functions.";
        goto fn_exit;
    }

    if (util_gpu_collection_kind() == GPU_COLLECTION_MIXED) {
        *pdisabled_reason = "No support for systems with mixed compute capabilities, such as CC < 7.0 and CC > 7.0 GPUS.";
        papi_errno = PAPI_ECMP;
        goto fn_exit;
    }
fn_exit:
    return papi_errno;
}

int util_runtime_is_perfworks_api(void)
{
    static int is_perfworks_api = -1;
    if (is_perfworks_api != -1)
        goto fn_exit;
    char *papi_cuda_110_cc70_perfworks_api = getenv("PAPI_CUDA_110_CC_70_PERFWORKS_API");

    enum gpu_collection_e gpus_kind = util_gpu_collection_kind();
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

int util_runtime_is_events_api(void)
{
    static int is_events_api = -1;
    if (is_events_api != -1)
        goto fn_exit;

    enum gpu_collection_e gpus_kind = util_gpu_collection_kind();

    /*
     * See lcuda_config.h: When NVIDIA removes the events API add a check in the following condition
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

int cucontext_array_create(void **pcuda_context)
{
    COMPDBG("Entering.\n");
    CUcontext *cuCtx = (CUcontext *) papi_calloc (get_device_count(), sizeof(CUcontext));
    if (cuCtx == NULL) {
        return PAPI_ENOMEM;
    }
    *pcuda_context = (void *) cuCtx;
    return PAPI_OK;
}

int cucontext_update_current(void *cuda_context)
{
    int papi_errno, gpu_id;
    CUcontext *cu_ctx = (CUcontext *) cuda_context;
    CUcontext tempCtx;
    papi_errno = cudaGetDevicePtr(&gpu_id);
    if (papi_errno != cudaSuccess) {
        return PAPI_EMISC;
    }
    papi_errno = cuCtxGetCurrentPtr(&tempCtx);
    if (papi_errno != CUDA_SUCCESS) {
        return PAPI_EMISC;
    }
    if (cu_ctx[gpu_id] == NULL) {
        if (tempCtx != NULL) {
            LOGDBG("Registering device = %d with ctx = %p.\n", gpu_id, tempCtx);
            CUDA_CALL(cuCtxGetCurrentPtr(&cu_ctx[gpu_id]), return PAPI_EMISC);
        }
        else {
            CUDART_CALL(cudaFreePtr(NULL), return PAPI_EMISC);
            CUDA_CALL(cuCtxGetCurrentPtr(&cu_ctx[gpu_id]), return PAPI_EMISC);
            LOGDBG("Using primary device context %p for device %d.\n", cu_ctx[gpu_id], gpu_id);
        }
    }
    else if (cu_ctx[gpu_id] != tempCtx) {  // If context has changed keep the first seen one but with warning
        ERRDBG("Warning: cuda context for gpu %d has changed from %p to %p\n", gpu_id, cu_ctx[gpu_id], tempCtx);
    }
    return PAPI_OK;
}

int cucontext_array_destroy(void **pcuda_context)
{
    papi_free(*pcuda_context);
    *pcuda_context = NULL;
    return PAPI_OK;
}
