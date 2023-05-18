#include <dlfcn.h>
#include <papi.h>
#include "papi_memory.h"

#include "lcuda_config.h"
#include "lcuda_utils.h"
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
    return PAPI_ENOSUPP;
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

static int load_cudart_sym(void)
{
    char dlname[] = "libcudart.so";
    char *found_files[MAX_FILES];
    int count, i, found = 0;
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root) {
        count = search_files_in_path(dlname, papi_cuda_root, found_files);
        for (i = 0; i < count; i++) {
            dl_rt = dlopen(found_files[i], RTLD_NOW | RTLD_GLOBAL);
            if (dl_rt) {
                found = 1;
                break;
            }
        }
        for (i = 0; i < count; i++) {
            free(found_files[i]);
        }
    }
    if (!found) {
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
    return PAPI_ENOSUPP;
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
    char *found_files[MAX_FILES];
    int count, i, found = 0;
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root) {
        count = search_files_in_path(dlname, papi_cuda_root, found_files);
        for (i = 0; i < count; i++) {
            dl_cupti = dlopen(found_files[i], RTLD_NOW | RTLD_GLOBAL);
            if (dl_cupti) {
                found = 1;
                break;
            }
        }
        for (i =0; i < count; i++) {
            free(found_files[i]);
        }
    }
    if (!found) {
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
    return PAPI_ENOSUPP;
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

int util_load_cuda_sym(const char **pdisabled_reason)
{
    int res;
    res = load_cuda_sym();
    res += load_cudart_sym();
    res += load_cupti_common_sym();
    if (res != PAPI_OK) {
        *pdisabled_reason = "Unable to load CUDA library functions.";
        return PAPI_ESYS;
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
    CUDART_CALL(cudaRuntimeGetVersionPtr(&runtimeVersion), return PAPI_ENOSUPP );
    return runtimeVersion;
}

static int util_dylib_cupti_version(void)
{
    unsigned int cuptiVersion;
    CUPTI_CALL(cuptiGetVersionPtr(&cuptiVersion), return PAPI_ENOSUPP );
    return cuptiVersion;
}

int get_device_count(void)
{
    int numDevs;
    CUDART_CALL(cudaGetDeviceCountPtr(&numDevs), return PAPI_ENOSUPP);
    return numDevs;
}

static int get_gpu_compute_capability(int dev_num)
{
    int cc_major, cc_minor;
    int cc;
    CUDART_CALL(cudaDeviceGetAttributePtr(&cc_major,
        cudaDevAttrComputeCapabilityMajor, dev_num),
            return PAPI_ENOSUPP );
    CUDART_CALL(cudaDeviceGetAttributePtr(&cc_minor,
        cudaDevAttrComputeCapabilityMinor, dev_num),
            return PAPI_ENOSUPP );
    cc = cc_major * 10 + cc_minor;
    return cc;
}

enum gpu_collection_e util_gpu_collection_kind(void)
{
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
        return GPU_COLLECTION_ALL_CC70;
    }
    if (count_perf == total_gpus) {
        return GPU_COLLECTION_ALL_PERF;
    }
    if (count_evt == total_gpus) {
        return GPU_COLLECTION_ALL_EVENTS;
    }
    return GPU_COLLECTION_MIXED;
}

int util_runtime_is_perfworks_api(void)
{
    enum gpu_collection_e gpus_kind = util_gpu_collection_kind();
    unsigned int cuptiVersion = util_dylib_cupti_version();

    if (gpus_kind == GPU_COLLECTION_ALL_CC70 && 
        (cuptiVersion == CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION || util_dylib_cu_runtime_version() == 11000))
    {
#if defined(PAPI_CUDA_110_CC_70_PERFWORKS_API)
        return 1;
#else
        return 0;
#endif
    }
    if ((gpus_kind == GPU_COLLECTION_ALL_PERF || gpus_kind == GPU_COLLECTION_ALL_CC70)
        && cuptiVersion >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION)
        return 1;
    else
        return 0;
}

int util_runtime_is_events_api(void)
{
    enum gpu_collection_e gpus_kind = util_gpu_collection_kind();
    if ((gpus_kind == GPU_COLLECTION_ALL_EVENTS || gpus_kind == GPU_COLLECTION_ALL_CC70)) 
        // TODO: && util_dylib_cupti_version() <= CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION)
        return 1;
    else
        return 0;
}

int cucontext_array_init(void **pcuda_context)
{
    COMPDBG("Entering.\n");
    CUcontext *cuCtx = (CUcontext *) papi_calloc (get_device_count(), sizeof(CUcontext));
    if (cuCtx == NULL) {
        return PAPI_ENOMEM;
    }
    *pcuda_context = (void *) cuCtx;
    return PAPI_OK;
}

int cucontext_array_free(void **pcuda_context)
{
    free(*pcuda_context);
    *pcuda_context = NULL;
    return PAPI_OK;
}
