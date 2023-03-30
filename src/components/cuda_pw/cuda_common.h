/*
    Load cuda symbols common to perfworks and events API
*/
#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#include <stdio.h>
#include "debug_comp.h"
#include "api_cuda.h"

char* PAPI_CUDA_ROOT_ENV;

void *dl1, *dl2, *dl3;

#define DECLARECUDAFUNC(funcname, funcsig) CUresult ( *funcname##Ptr ) funcsig;
DECLARECUDAFUNC(cuCtxGetCurrent, (CUcontext *));
DECLARECUDAFUNC(cuCtxSetCurrent, (CUcontext));
DECLARECUDAFUNC(cuCtxDestroy, (CUcontext));
DECLARECUDAFUNC(cuCtxCreate, (CUcontext *pctx, unsigned int flags, CUdevice dev));
DECLARECUDAFUNC(cuCtxGetDevice, (CUdevice *));
DECLARECUDAFUNC(cuDeviceGet, (CUdevice *, int));
DECLARECUDAFUNC(cuDeviceGetCount, (int *));
DECLARECUDAFUNC(cuDeviceGetName, (char *, int, CUdevice));
DECLARECUDAFUNC(cuDevicePrimaryCtxRetain, (CUcontext *pctx, CUdevice));
DECLARECUDAFUNC(cuDevicePrimaryCtxRelease, (CUdevice));
DECLARECUDAFUNC(cuInit, (unsigned int));
DECLARECUDAFUNC(cuGetErrorString, (CUresult error, const char** pStr));
DECLARECUDAFUNC(cuCtxPopCurrent, (CUcontext * pctx));
DECLARECUDAFUNC(cuCtxPushCurrent, (CUcontext pctx));
DECLARECUDAFUNC(cuCtxSynchronize, ());
DECLARECUDAFUNC(cuDeviceGetAttribute, (int *, CUdevice_attribute, CUdevice));

#define DECLARECUDARTFUNC(funcname, funcsig) cudaError_t ( *funcname##Ptr ) funcsig;
DECLARECUDARTFUNC(cudaGetDeviceCount, (int *));
DECLARECUDARTFUNC(cudaGetDevice, (int *));
DECLARECUDARTFUNC(cudaSetDevice, (int));
DECLARECUDARTFUNC(cudaGetDeviceProperties, (struct cudaDeviceProp* prop, int  device));
DECLARECUDARTFUNC(cudaDeviceGetAttribute, (int *value, enum cudaDeviceAttr attr, int device));
DECLARECUDARTFUNC(cudaFree, (void *));
DECLARECUDARTFUNC(cudaDriverGetVersion, (int *));
DECLARECUDARTFUNC(cudaRuntimeGetVersion, (int *));

#define DECLARECUPTIFUNC(funcname, funcsig) CUptiResult ( *funcname##Ptr ) funcsig;
DECLARECUPTIFUNC(cuptiGetVersion, (uint32_t* ));

#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name );  \
    if (dlerror() != NULL) {  \
        fprintf(stderr, "A CUDA required function '%s' was not found in lib '%s'.\n", name, #dllib);  \
        return PAPI_ENOSUPP;  \
    }

#define CUDA_CALL( call, handleerror )  \
    do {  \
        CUresult _status = (call);  \
        LOGCUDACALL("%s\t" #call "\n", __func__);  \
        if (_status != CUDA_SUCCESS) {  \
            fprintf(stderr, "\033[31;1mCUDA Error %d: %s: %d: Error in call to " #call "\033[0m\n", _status, __FILE__, __LINE__);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);
#define CUDART_CALL( call, handleerror )  \
    do {  \
        cudaError_t _status = (call);  \
        LOGCUDACALL("%s\t" #call "\n", __func__);  \
        if (_status != cudaSuccess) {  \
            fprintf(stderr, "\033[31;1mCUDART Error %d: %s: %d: Error in call to " #call "\033[0m\n", _status, __FILE__, __LINE__);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);
#define CUPTI_CALL( call, handleerror ) \
    do {  \
        CUptiResult _status = (call);  \
        LOGCUPTICALL("%s\t" #call "\033[0m\n", __func__);  \
        if (_status != CUPTI_SUCCESS) {  \
            fprintf(stderr, "\033[31;1mCUPTI Error %d: %s: %d: Error in call to " #call "\033[0m\n", _status, __FILE__, __LINE__);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

int get_env_PAPI_CUDA_ROOT(void);
int load_cuda_sym(void);
int load_cudart_sym(void);
int load_cupti_common_sym(void);

int check_cuda_api_versions(void);
int get_device_count(void);
int is_gpu_perfworks(int dev_num);
int is_mixed_compute_capability(void);
int init_CUcontext_array(void ** pcuda_context);

#endif /* __CUDA_COMMON_H__ */