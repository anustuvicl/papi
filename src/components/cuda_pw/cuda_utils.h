/*
    Load cuda symbols common to perfworks and events API
*/
#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>
#include "debug_comp.h"
#include "cuda_api_config.h"

char* PAPI_CUDA_ROOT_ENV;

void *dl1, *dl2, *dl3;

// cuda driver function pointers
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

// cuda runtime function pointers
cudaError_t ( *cudaGetDeviceCountPtr ) (int *);
cudaError_t ( *cudaGetDevicePtr ) (int *);
cudaError_t ( *cudaSetDevicePtr ) (int);
cudaError_t ( *cudaGetDevicePropertiesPtr ) (struct cudaDeviceProp* prop, int  device);
cudaError_t ( *cudaDeviceGetAttributePtr ) (int *value, enum cudaDeviceAttr attr, int device);
cudaError_t ( *cudaFreePtr ) (void *);
cudaError_t ( *cudaDriverGetVersionPtr ) (int *);
cudaError_t ( *cudaRuntimeGetVersionPtr ) (int *);

CUptiResult ( *cuptiGetVersionPtr ) (uint32_t* );

#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name );  \
    if (dlerror() != NULL) {  \
        fprintf(stderr, "A CUDA required function '%s' was not found in lib '%s'.\n", name, #dllib);  \
        return PAPI_ENOSUPP;  \
    }

#define CUDA_CALL( call, handleerror )  \
    do {  \
        CUresult _status = (call);  \
        LOGCUDACALL("\t" #call "\n");  \
        if (_status != CUDA_SUCCESS) {  \
            fprintf(stderr, "\033[31;1mCUDA Error %d: %s: %d: Error in call to " #call "\033[0m\n", _status, __FILE__, __LINE__);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);
#define CUDART_CALL( call, handleerror )  \
    do {  \
        cudaError_t _status = (call);  \
        LOGCUDACALL("\t" #call "\n");  \
        if (_status != cudaSuccess) {  \
            fprintf(stderr, "\033[31;1mCUDART Error %d: %s: %d: Error in call to " #call "\033[0m\n", _status, __FILE__, __LINE__);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);
#define CUPTI_CALL( call, handleerror ) \
    do {  \
        CUptiResult _status = (call);  \
        LOGCUPTICALL("\t" #call "\033[0m\n");  \
        if (_status != CUPTI_SUCCESS) {  \
            fprintf(stderr, "\033[31;1mCUPTI Error %d: %s: %d: Error in call to " #call "\033[0m\n", _status, __FILE__, __LINE__);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

int get_env_papi_cuda_root(void);
int load_cuda_sym(void);
int load_cudart_sym(void);
int load_cupti_common_sym(void);

int check_cuda_api_versions(void);
int get_device_count(void);
int is_gpu_perfworks(int dev_num);
int is_mixed_compute_capability(void);
int init_CUcontext_array(void ** pcuda_context);

#endif /* __CUDA_UTILS_H__ */