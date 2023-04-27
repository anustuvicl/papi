NAME=cuda_pw
include ../../Makefile_comp_tests.target

PAPI_CUDA_ROOT ?= /opt/cuda
PAPI_CUPTI_ROOT ?= $(PAPI_CUDA_ROOT)/extras/CUPTI

TESTS = HelloWorld simpleMultiGPU \
		pthreads cudaOpenMP test_cuda_component concurrent_profiling \
		test_multi_read test_multipass_event_fail

TESTS_NOCTX = concurrent_profiling_noCuCtx pthreads_noCuCtx \
			  cudaOpenMP_noCuCtx HelloWorld_noCuCtx \
			  simpleMultiGPU_noCuCtx

cuda_pw_tests: $(TESTS) $(TESTS_NOCTX)

NVCC = $(PAPI_CUDA_ROOT)/bin/nvcc
NVCFLAGS = -g -ccbin='$(CC)'
INCLUDE += -I$(PAPI_CUDA_ROOT)/include -I$(PAPI_CUPTI_ROOT)/include
CUDALIBS = -L$(CUDRV_DIR)/lib64 -L$(PAPI_CUDA_ROOT)/lib64 -L$(PAPI_CUDA_ROOT)/lib64/stubs -L$(PAPI_CUPTI_ROOT)/lib64 -lcudart -lcupti -lcuda
PAPILIB += -L../../../libpfm4/lib -lpfm
default: $(TESTS)

%.o:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c -o $@ $<

%.mac:%.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -E -c -o $@ $<

test_cuda_component: test_cuda_component.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_cuda_component test_cuda_component.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

test_multi_read: test_multi_read.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_multi_read test_multi_read.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling: concurrent_profiling.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o concurrent_profiling concurrent_profiling.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling_noCuCtx: concurrent_profiling_noCuCtx.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o concurrent_profiling_noCuCtx concurrent_profiling_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads: pthreads.o
	$(NVCC) $(NVCFLAGS) -o pthreads pthreads.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads_noCuCtx: pthreads_noCuCtx.o
	$(NVCC) $(NVCFLAGS) -o pthreads_noCuCtx pthreads_noCuCtx.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP: cudaOpenMP.o
	$(NVCC) $(NVCFLAGS) -o cudaOpenMP cudaOpenMP.o -lgomp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP.o: cudaOpenMP.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c cudaOpenMP.cu -Xcompiler -fopenmp

cudaOpenMP_noCuCtx: cudaOpenMP_noCuCtx.o
	$(NVCC) $(NVCFLAGS) -o cudaOpenMP_noCuCtx cudaOpenMP_noCuCtx.o -lgomp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP_noCuCtx.o: cudaOpenMP_noCuCtx.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c cudaOpenMP_noCuCtx.cu -Xcompiler -fopenmp

test_multipass_event_fail: test_multipass_event_fail.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_multipass_event_fail test_multipass_event_fail.o $(INCLUDE) $(UTILOBJS) $(PAPILIB) $(LDFLAGS) $(CUDALIBS)

test_2thr_1gpu_not_allowed: test_2thr_1gpu_not_allowed.o
	$(NVCC) $(NVCFLAGS) -o test_2thr_1gpu_not_allowed test_2thr_1gpu_not_allowed.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld: HelloWorld.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld HelloWorld.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

HelloWorld_noCuCtx: HelloWorld_noCuCtx.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld_noCuCtx HelloWorld_noCuCtx.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU: simpleMultiGPU.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_noCuCtx: simpleMultiGPU_noCuCtx.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

clean:
	rm -f *.o $(TESTS) $(TESTS_NOCTX)
