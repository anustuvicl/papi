#include <stdio.h>
#include <dlfcn.h>
#include <papi.h>
#include "papi_memory.h"

#include "cuda_utils.h"
#include "common.h"

static void *dl_drv, *dl_rt;

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

int utils_load_cuda_sym(void)
{
    int res;
    res = load_cuda_sym();
    res += load_cudart_sym();
    res += load_cupti_common_sym();
    if (res != PAPI_OK)
        return PAPI_ESYS;
    else
        return PAPI_OK;
}

int utils_unload_cuda_sym(void)
{
    unload_cuda_sym();
    unload_cudart_sym();
    unload_cupti_common_sym();
    return PAPI_OK;
}

int check_cuda_api_versions(void)
{
    int runtimeversion, driverversion;
    unsigned int cuptiversion;
    CUDART_CALL(cudaRuntimeGetVersionPtr(&runtimeversion), return PAPI_ENOSUPP );
    CUDART_CALL(cudaDriverGetVersionPtr(&driverversion), return PAPI_ENOSUPP );
    CUPTI_CALL(cuptiGetVersionPtr(&cuptiversion), return PAPI_ENOSUPP );
    // if (runtimeversion != CUDA_VERSION || cuptiversion != CUPTI_API_VERSION)
        // return PAPI_ECMP;
    if (driverversion < runtimeversion)
        return PAPI_ESYS;
    return PAPI_OK;
}

int get_device_count(void)
{
    int numDevs;
    CUDART_CALL(cudaGetDeviceCountPtr(&numDevs), return PAPI_ENOSUPP);
    return numDevs;
}

int is_gpu_perfworks(int dev_num)
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
    if (cc >= 70)
        return PAPI_OK;
    else
        return PAPI_ENOSUPP;
}

int is_mixed_compute_capability(void)
{
    int total_gpus = get_device_count();
    int i, count=0;
    for (i=0; i<total_gpus; i++) {
        if (is_gpu_perfworks(i) == PAPI_OK)
            count++;
    }
    if (count == total_gpus)
        return PAPI_OK;
    else
        return PAPI_ENOSUPP;
}

int CUcontext_array_init(void ** pcuda_context)
{
    COMPDBG("Entering.\n");
    CUcontext *cuCtx = (CUcontext *) papi_calloc (get_device_count(), sizeof(CUcontext));
    if (cuCtx == NULL) {
        return PAPI_ENOMEM;
    }
    *pcuda_context = (void *) cuCtx;
    return PAPI_OK;
}

int CUcontext_array_free(void **pcuda_context)
{
    free(*pcuda_context);
    *pcuda_context = NULL;
    return PAPI_OK;
}
