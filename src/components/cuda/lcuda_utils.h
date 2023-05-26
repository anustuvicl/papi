/**
 * @file    lcuda_utils.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>

#include "lcuda_debug.h"

extern void *dl_cupti;

// cuda driver function pointers
extern CUresult ( *cuCtxGetCurrentPtr ) (CUcontext *);
extern CUresult ( *cuCtxSetCurrentPtr ) (CUcontext);
extern CUresult ( *cuCtxDestroyPtr ) (CUcontext);
extern CUresult ( *cuCtxCreatePtr ) (CUcontext *pctx, unsigned int flags, CUdevice dev);
extern CUresult ( *cuCtxGetDevicePtr ) (CUdevice *);
extern CUresult ( *cuDeviceGetPtr ) (CUdevice *, int);
extern CUresult ( *cuDeviceGetCountPtr ) (int *);
extern CUresult ( *cuDeviceGetNamePtr ) (char *, int, CUdevice);
extern CUresult ( *cuDevicePrimaryCtxRetainPtr ) (CUcontext *pctx, CUdevice);
extern CUresult ( *cuDevicePrimaryCtxReleasePtr ) (CUdevice);
extern CUresult ( *cuInitPtr ) (unsigned int);
extern CUresult ( *cuGetErrorStringPtr ) (CUresult error, const char** pStr);
extern CUresult ( *cuCtxPopCurrentPtr ) (CUcontext * pctx);
extern CUresult ( *cuCtxPushCurrentPtr ) (CUcontext pctx);
extern CUresult ( *cuCtxSynchronizePtr ) ();
extern CUresult ( *cuDeviceGetAttributePtr ) (int *, CUdevice_attribute, CUdevice);

// cuda runtime function pointers
extern cudaError_t ( *cudaGetDeviceCountPtr ) (int *);
extern cudaError_t ( *cudaGetDevicePtr ) (int *);
extern cudaError_t ( *cudaSetDevicePtr ) (int);
extern cudaError_t ( *cudaGetDevicePropertiesPtr ) (struct cudaDeviceProp* prop, int  device);
extern cudaError_t ( *cudaDeviceGetAttributePtr ) (int *value, enum cudaDeviceAttr attr, int device);
extern cudaError_t ( *cudaFreePtr ) (void *);
extern cudaError_t ( *cudaDriverGetVersionPtr ) (int *);
extern cudaError_t ( *cudaRuntimeGetVersionPtr ) (int *);

extern CUptiResult ( *cuptiGetVersionPtr ) (uint32_t* );

#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name );  \
    if (dlerror() != NULL) {  \
        ERRDBG("A CUDA required function '%s' was not found in lib '%s'.\n", name, #dllib);  \
        return PAPI_ESYS;  \
    }

#define CUDA_CALL( call, handleerror )  \
    do {  \
        CUresult _status = (call);  \
        LOGCUDACALL("\t" #call "\n");  \
        if (_status != CUDA_SUCCESS) {  \
            ERRDBG("CUDA Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);
#define CUDART_CALL( call, handleerror )  \
    do {  \
        cudaError_t _status = (call);  \
        LOGCUDACALL("\t" #call "\n");  \
        if (_status != cudaSuccess) {  \
            ERRDBG("CUDART Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);
#define CUPTI_CALL( call, handleerror ) \
    do {  \
        CUptiResult _status = (call);  \
        LOGCUPTICALL("\t" #call "\n");  \
        if (_status != CUPTI_SUCCESS) {  \
            ERRDBG("CUPTI Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

int util_load_cuda_sym(const char **pdisabled_reason);
int util_unload_cuda_sym(void);

enum gpu_collection_e {GPU_COLLECTION_UNKNOWN, GPU_COLLECTION_ALL_PERF, GPU_COLLECTION_MIXED, GPU_COLLECTION_ALL_EVENTS, GPU_COLLECTION_ALL_CC70};

int get_device_count(void);
enum gpu_collection_e util_gpu_collection_kind(void);
int util_runtime_is_perfworks_api(void);
int util_runtime_is_events_api(void);

int cucontext_array_init(void **pcuda_context);
int cucontext_array_free(void **pcuda_context);

#endif /* __CUDA_UTILS_H__ */
