#include <stdio.h>
#include <dlfcn.h>
#include <papi.h>

#include "cuda_common.h"

int get_env_PAPI_CUDA_ROOT(void)
{
    PAPI_CUDA_ROOT_ENV = getenv("PAPI_CUDA_ROOT");
    if (PAPI_CUDA_ROOT_ENV == NULL) {
        return PAPI_ECMP;
    }
    return PAPI_OK;
}

int load_cuda_sym(void)
{
    int papiErr = PAPI_OK;
    // char path_lib[PATH_MAX];
    // int strErr;
    // strErr = snprintf(path_lib, PATH_MAX-2, "%s/lib64/libcuda.so", PAPI_CUDA_ROOT);
    // printf("path_lib: %s\n", path_lib);

    // $PAPI_CUDA_ROOT/targets/x86_64-linux/lib/stubs/libcuda.so
    dl1 = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (dl1 == NULL) {
        fprintf(stderr, "ERROR: Loading libcuda.so failed.\n");
        goto fn_fail;
    }

    cuCtxSetCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxSetCurrent");
    cuCtxGetCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxGetCurrent");
    cuCtxDestroyPtr = DLSYM_AND_CHECK(dl1, "cuCtxDestroy");
    cuCtxCreatePtr = DLSYM_AND_CHECK(dl1, "cuCtxCreate");
    cuCtxGetDevicePtr = DLSYM_AND_CHECK(dl1, "cuCtxGetDevice");
    cuDeviceGetPtr = DLSYM_AND_CHECK(dl1, "cuDeviceGet");
    cuDeviceGetCountPtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetCount");
    cuDeviceGetNamePtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetName");
    cuDevicePrimaryCtxRetainPtr = DLSYM_AND_CHECK(dl1, "cuDevicePrimaryCtxRetain");
    cuDevicePrimaryCtxReleasePtr = DLSYM_AND_CHECK(dl1, "cuDevicePrimaryCtxRelease");
    cuInitPtr = DLSYM_AND_CHECK(dl1, "cuInit");
    cuGetErrorStringPtr = DLSYM_AND_CHECK(dl1, "cuGetErrorString");
    cuCtxPopCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr = DLSYM_AND_CHECK(dl1, "cuCtxPushCurrent");
    cuCtxSynchronizePtr = DLSYM_AND_CHECK(dl1, "cuCtxSynchronize");
    cuDeviceGetAttributePtr = DLSYM_AND_CHECK(dl1, "cuDeviceGetAttribute");

    fn_exit:
        return papiErr;
    fn_fail:
        papiErr = PAPI_ENOSUPP;
        goto fn_exit;
}

int load_cudart_sym(void)
{
    int papiErr = PAPI_OK;
    dl2 = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    if (dl2 == NULL) {
        fprintf(stderr, "ERROR: Loading libcudart.so failed.\n");
        goto fn_fail;
    }

    cudaGetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaGetDevice");
    cudaGetDeviceCountPtr = DLSYM_AND_CHECK(dl2, "cudaGetDeviceCount");
    cudaGetDevicePropertiesPtr = DLSYM_AND_CHECK(dl2, "cudaGetDeviceProperties");
    cudaDeviceGetAttributePtr = DLSYM_AND_CHECK(dl2, "cudaDeviceGetAttribute");
    cudaSetDevicePtr = DLSYM_AND_CHECK(dl2, "cudaSetDevice");
    cudaFreePtr = DLSYM_AND_CHECK(dl2, "cudaFree");
    cudaDriverGetVersionPtr = DLSYM_AND_CHECK(dl2, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr = DLSYM_AND_CHECK(dl2, "cudaRuntimeGetVersion");

    fn_exit:
        return papiErr;
    fn_fail:
        papiErr = PAPI_ENOSUPP;
        goto fn_exit;
}

int load_cupti_common_sym(void)
{
    int papiErr = PAPI_OK;
    dl3 = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    if (dl3 == NULL) {
        fprintf(stderr, "ERROR: Loading libcupti.so failed.\n");
        goto fn_fail;
    }

    cuptiGetVersionPtr = DLSYM_AND_CHECK(dl3, "cuptiGetVersion");

    fn_exit:
        return papiErr;
    fn_fail:
        papiErr = PAPI_ENOSUPP;
        goto fn_exit;
}

int check_cuda_api_versions(void)
{
    int runtimeversion, driverversion;
    unsigned int cuptiversion;
    CUDART_CALL(cudaRuntimeGetVersionPtr(&runtimeversion), return PAPI_ENOSUPP );
    CUDART_CALL(cudaDriverGetVersionPtr(&driverversion), return PAPI_ENOSUPP );
    CUPTI_CALL(cuptiGetVersionPtr(&cuptiversion), return PAPI_ENOSUPP );
    if (runtimeversion != CUDA_VERSION || cuptiversion != CUPTI_API_VERSION)
        return PAPI_ECMP;
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

int init_CUcontext_array(void ** pcuda_context)
{
    COMPDBG("Entering.\n");
    CUcontext *cuCtx = (CUcontext *) calloc (get_device_count(), sizeof(CUcontext));
    if (cuCtx == NULL) {
        return PAPI_ENOMEM;
    }
    *pcuda_context = (void *) cuCtx;
    return PAPI_OK;
}

// int get_gpu_info(void) {
//     COMPDBG("Entering.\n");
//     int i;
//     for (i=0; i<get_device_count(); i++) {
//         CUDART_CALL(cudaGetDevicePropertiesPtr(&_devprop, i));
//         strcpy(devinfo[i].deviceName, _devprop.name);
//         }
// }
