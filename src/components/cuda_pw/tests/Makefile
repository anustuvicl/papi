NAME=cuda_pw
include ../../Makefile_comp_tests.target

PAPI_CUDA_ROOT ?= /opt/cuda
PAPI_CUPTI_ROOT ?= $(PAPI_CUDA_ROOT)/extras/CUPTI

TESTS = HelloWorld_prof simpleMultiGPU_CUPTI11 pthreads cudaOpenMP test_cuda_component test_multi_read concurrent_profiling
cuda_pw_tests: $(TESTS)

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

list_metrics: list_metrics.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o list_metrics list_metrics.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

name2code: name2code.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o name2code name2code.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

test_cuda_component: test_cuda_component.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_cuda_component test_cuda_component.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

test_multi_read: test_multi_read.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_multi_read test_multi_read.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

concurrent_profiling: concurrent_profiling.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o concurrent_profiling concurrent_profiling.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

pthreads: pthreads.o
	$(NVCC) $(NVCFLAGS) -o pthreads pthreads.o -lpthread $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP: cudaOpenMP.o
	$(NVCC) $(NVCFLAGS) -o cudaOpenMP cudaOpenMP.o -lgomp $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

cudaOpenMP.o: cudaOpenMP.cu
	$(NVCC) $(INCLUDE) $(NVCFLAGS) -c cudaOpenMP.cu -Xcompiler -fopenmp

test_multipass_event_fail: test_multipass_event_fail.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o test_multipass_event_fail test_multipass_event_fail.o $(INCLUDE) $(UTILOBJS) $(PAPILIB) $(LDFLAGS) $(CUDALIBS)

HelloWorld_prof: HelloWorld_prof.o $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) -o HelloWorld_prof HelloWorld_prof.o $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

simpleMultiGPU_CUPTI11: simpleMultiGPU_CUPTI11.cu $(UTILOBJS)
	$(NVCC) $(NVCFLAGS) $(INCLUDE) -DPAPI -g -o $@ $+ $(UTILOBJS) $(PAPILIB) $(CUDALIBS) $(LDFLAGS)

clean:
	rm -f *.o $(TESTS)
