// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This sample demonstrates two ways to use the CUPTI Profiler API with concurrent kernels.
// By taking the ratio of runtimes for a consecutive series of kernels, compared
// to a series of concurrent kernels, one can difinitively demonstrate that concurrent
// kernels were running while metrics were gathered and the User Replay mechanism was in use.
//
// Example:
// 4 kernel launches, with 1x, 2x, 3x, and 4x amounts of work, each sized to one SM (one warp
// of threads, one thread block).
// When run synchronously, this comes to 10x amount of work.
// When run concurrently, the longest (4x) kernel should be the only measured time (it hides the others).
// Thus w/ 4 kernels, the concurrent : consecutive time ratio should be 4:10.
// On test hardware this does simplify to 3.998:10.  As the test is affected by memory layout, this may not
// hold for certain architectures where, for example, cache sizes may optimize certain kernel calls.
//
// After demonstrating concurrency using multpile streams, this then demonstrates using multiple devices.
// In this 3rd configuration, the same concurrent workload with streams is then duplicated and run
// on each device concurrently using streams.
// In this case, the wallclock time to launch, run, and join the threads should be roughly the same as the
// wallclock time to run the single device case.  If concurrency was not working, the wallcock time
// would be (num devices) times the single device concurrent case.
//
//  * If the multiple devices have different performance, the runtime may be significantly different between
//    devices, but this does not mean concurrent profiling is not happening.

// This code has been adapted to PAPI from 
// `<CUDA-TOOLKIT-11.4>/extras/CUPTI/samples/concurrent_profiling/cpncurrent_profiling.cu`

extern "C" {
    #include <papi.h>
}
// Standard CUDA, CUPTI, Profiler, NVPW headers
#include "cuda.h"

// Standard STL headers
#include <chrono>
#include <cstdint>
#include <iostream>

using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <string>
using ::std::string;

#include <thread>
using ::std::thread;

#include <vector>
using ::std::vector;

#define PAPI_CALL(apiFuncCall)                                          \
do {                                                                           \
    int _status = apiFuncCall;                                         \
    if (_status != PAPI_OK) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d(%s).\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status, PAPI_strerror(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

// Helpful error handlers for standard CUPTI and CUDA runtime calls
#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

typedef struct
{
    int device;                //!< compute device number
    CUcontext context;         //!< CUDA driver context, or NULL if default context has already been initialized
} profilingConfig;

// Per-device configuration, buffers, stream and device information, and device pointers
typedef struct
{
    int deviceID;
    profilingConfig config;                 // Each device (or each context) needs its own CUPTI profiling config
    vector<cudaStream_t> streams;           // Each device needs its own streams
    vector<double *> d_x;                   // And device memory allocation
    vector<double *> d_y;                   // ..
    long long values[100];                  // Capture PAPI measured values for each device
} perDeviceData;

#define DAXPY_REPEAT 32768
// Loop over array of elements performing daxpy multiple times
// To be launched with only one block (artificially increasing serial time to better demonstrate overlapping replay)
__global__ void daxpyKernel(int elements, double a, double * x, double * y)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        // Artificially increase kernel runtime to emphasize concurrency
        for (int j = 0; j < DAXPY_REPEAT; j++)
            y[i] = a * x[i] + y[i]; // daxpy
}

// Initialize kernel values
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices
// we run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels (streams, when running concurrently)
int const numKernels = 4;
int const numStreams = numKernels;
vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB
int const blockSize = 4 * 1024;

// Wrapper which will launch numKernel kernel calls on a single device
// The device streams vector is used to control which stream each call is made on
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used
void profileKernels(perDeviceData &d,
                    vector<string> const &metricNames,
                    char const * const rangeName, bool serial)
{
    int eventset = PAPI_NULL, i;
    PAPI_CALL(PAPI_create_eventset(&eventset));
    // Switch to desired device
    RUNTIME_API_CALL(cudaSetDevice(d.config.device));  // Orig code has mistake here
    DRIVER_API_CALL(cuCtxSetCurrent(d.config.context));
    string evt_name;
    for (i = 0; i < metricNames.size(); i++) {
        evt_name = metricNames[i] + std::to_string(d.config.device);
        cout<<"Adding event name: " << evt_name << endl;
        PAPI_CALL(PAPI_add_named_event(eventset, evt_name.c_str()));
    }
    PAPI_CALL(PAPI_start(eventset));

        for (unsigned int stream = 0; stream < d.streams.size(); stream++)
        {
            cudaStream_t streamId = (serial ? 0 : d.streams[stream]);
            daxpyKernel <<<threadBlocks, threadsPerBlock, 0, streamId>>> (elements[stream], a, d.d_x[stream], d.d_y[stream]);
        }

        // After launching all work, synchronize all streams
        if (serial == false)
        {
            for (unsigned int stream = 0; stream < d.streams.size(); stream++)
            {
                RUNTIME_API_CALL(cudaStreamSynchronize(d.streams[stream]));
            }
        }
        else
        {
            RUNTIME_API_CALL(cudaStreamSynchronize(0));
        }
    PAPI_CALL(PAPI_stop(eventset, d.values));
    PAPI_CALL(PAPI_cleanup_eventset(eventset));
}

void print_measured_values(perDeviceData &d, vector<string> const &metricNames)
{
    string evt_name;
    cout << "PAPI event name" << std::string(7, '\t') << "Measured value" << endl;
    cout << std::string(80, '-') << endl;
    for (int i=0; i < metricNames.size(); i++) {
        evt_name = metricNames[i] + std::to_string(d.config.device);
        cout << evt_name << "\t\t\t" << d.values[i] << endl;
    }
}

int main(int argc, char * argv[])
{
    // These two metrics will demonstrate whether kernels within a Range were run serially or concurrently
    vector<string> metricNames;
    metricNames.push_back("cuda_pw:::sm__cycles_active.sum:device=");
    metricNames.push_back("cuda_pw:::sm__cycles_elapsed.max:device=");
    // This metric shows that the same number of flops were executed on each run
    // Note: PAPI can't measure this with the others as it requires multiple passes to measure as three
    //       However, it can be measured as a single event.
    // metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum");

    // Initialize the PAPI library
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init failed.\n");
        exit(-1);
    }

    int numDevices;
    RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));

    // Per-device information
    vector<int> device_ids;

    // Find all devices capable of running CUPTI Profiling (Compute Capability >= 7.0)
    for (int i = 0; i < numDevices; i++)
    {
        // Get device properties
        int major;
        RUNTIME_API_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i));
        if (major >= 7)
        {
            // Record device number
            device_ids.push_back(i);
        }
    }

    numDevices = device_ids.size();
    cout << "Found " << numDevices << " compatible devices" << endl;

    // Ensure we found at least one device
    if (numDevices == 0)
    {
        cerr << "No devices detected compatible with CUPTI Profiling (Compute Capability >= 7.0)" << endl;
        exit(-1);
    }

    // Initialize kernel input to some known numbers
    vector<double> h_x(blockSize * numKernels);
    vector<double> h_y(blockSize * numKernels);
    for (size_t i = 0; i < blockSize * numKernels; i++)
    {
        h_x[i] = 1.5 * i;
        h_y[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels
    vector<cudaStream_t> defaultStreams(numStreams);
    for (int stream = 0; stream < numStreams; stream++)
    {
        defaultStreams[stream] = 0;
    }

    // Scale per-kernel work by stream number
    for (int stream = 0; stream < numStreams; stream++)
    {
        elements[stream] = blockSize * (stream + 1);
    }

    // For each device, configure profiling, set up buffers, copy kernel data
    vector<perDeviceData> deviceData(numDevices);

    for (int device = 0; device < numDevices; device++)
    {
        int device_id = device_ids[device];
        RUNTIME_API_CALL(cudaSetDevice(device_id));
        cout << "Configuring device " << device_id << endl;

        // Required CUPTI Profiling configuration & initialization
        // Can be done ahead of time or immediately before startSession() call
        // Initialization & configuration images can be generated separately, then passed to later calls
        // For simplicity's sake, in this sample, a single config struct is created per device and passed to each CUPTI Profiler API call
        // For more complex cases, each combination of CUPTI Profiler Session and Config requires additional initialization
        profilingConfig config;
        config.device = device_id;         // Device ID, used to get device name for metrics enumeration
        // config.maxLaunchesPerPass = 1;     // Must be >= maxRangesPerPass.  Set this to the largest count of kernel launches which may be encountered in any Pass in this Session

        // // Device 0 has max of 3 passes; other devices only run one pass in this sample code
        // if (device == 0)
        // {
        //     config.maxNumRanges = 3;       // Maximum number of ranges that may be profiled in the current Session
        // }
        // else
        // {
        //     config.maxNumRanges = 1;       // Maximum number of ranges that may be profiled in the current Session
        // }

        // config.maxRangeNameLength = 64;    // Max length including NULL terminator of any range name
        // config.maxRangesPerPass = 1;       // Max ranges that can be recorded in any Pass in this Session
        // config.minNestingLevels = 1;       // Must be >= 1, minimum reported nest level for Ranges in this Session
        // config.numNestingLevels = 1;       // Must be >= 1, max height of nested Ranges in this Session
        // config.rangeMode = CUPTI_UserRange;// CUPTI_AutoRange or CUPTI_UserRange
        // config.replayMode = CUPTI_UserReplay; // CUPTI_KernelReplay, CUPTI_UserReplay, or CUPTI_ApplicationReplay
        DRIVER_API_CALL(cuCtxCreate(&(config.context), 0, device)); // Either set to a context, or may be NULL if a default context has been created
        deviceData[device].config = config;// Save this device config

        // Initialize CUPTI Profiling structures
        // targetInitProfiling(deviceData[device], metricNames);

        // Per-stream initialization & memory allocation - copy from constant host array to each device array
        deviceData[device].streams.resize(numStreams);
        deviceData[device].d_x.resize(numStreams);
        deviceData[device].d_y.resize(numStreams);
        for (int stream = 0; stream < numStreams; stream++)
        {
            RUNTIME_API_CALL(cudaStreamCreate(&(deviceData[device].streams[stream])));

            // Each kernel does (stream #) * blockSize work on doubles
            size_t size = elements[stream] * sizeof(double);

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_x[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_x[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_x[stream], h_x.data(), size, cudaMemcpyHostToDevice));

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_y[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_y[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_y[stream], h_x.data(), size, cudaMemcpyHostToDevice));
        }
    }

    //
    // First version - single device, kernel calls serialized on default stream
    //

    // Use wallclock time to measure performance
    auto begin_time = ::std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams, which run serially
    profileKernels(deviceData[0], metricNames, "single_device_serial", true);

    auto end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    int numBlocks = 0;
    for (int i = 1; i <= numKernels; i++)
    {
        numBlocks += i;
    }
    cout << "It took " << elapsed_serial_ms.count() << "ms on the host to profile " << numKernels << " kernels in serial." << endl;

    //
    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency
    // (Should be limited by the longest running kernel)
    //

    begin_time = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism
    profileKernels(deviceData[0], metricNames, "single_device_async", false);

    end_time = ::std::chrono::high_resolution_clock::now();
    auto elapsed_single_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
    cout << "It took " << elapsed_single_device_ms.count() << "ms on the host to profile " << numKernels << " kernels on a single device on separate streams." << endl;
    cout << "--> If the separate stream wallclock time is less than the serial version, the streams were profiling concurrently." << endl;

    //
    // Third version - same as the second case, but duplicates the concurrent work across devices to show cross-device concurrency
    // This is done using devices so no serialization is needed between devices
    // (Should have roughly the same wallclock time as second case if the devices have similar performance)
    //

    if (numDevices == 1)
    {
        cout << "Only one compatible device found; skipping the multi-threaded test." << endl;
    }
    else
    {
        if ( PAPI_OK != PAPI_thread_init((unsigned long (*)(void)) std::this_thread::get_id) ) {
            fprintf(stderr, "Error setting thread id function.\n");
            exit(-1);
        }

        cout << "Running on " << numDevices << " devices, one thread per device." << endl;

        // Time creation of the same multiple streams (on multiple devices, if possible)
        vector<::std::thread> threads;
        begin_time = ::std::chrono::high_resolution_clock::now();

        // Now launch parallel thread work, duplicated on one thread per device
        for (int thread = 0; thread < numDevices; thread++)
        {
            threads.push_back(::std::thread(profileKernels, ::std::ref(deviceData[thread]), metricNames, "multi_device_async", false));
        }

        // Wait for all threads to finish
        for (auto &t: threads)
        {
            t.join();
        }

        // Record time used when launching on multiple devices
        end_time = ::std::chrono::high_resolution_clock::now();
        auto elapsed_multiple_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time);
        cout << "It took " << elapsed_multiple_device_ms.count() << "ms on the host to profile the same " << numKernels << " kernels on each of the " << numDevices << " devices in parallel" << endl;
        cout << "--> Wallclock ratio of parallel device launch to single device launch is " << elapsed_multiple_device_ms.count() / static_cast<double>(elapsed_single_device_ms.count()) << endl;
        cout << "--> If the ratio is close to 1, that means there was little overhead to profile in parallel on multiple devices compared to profiling on a single device." << endl;
        cout << "--> If the devices have different performance, the ratio may not be close to one, and this should be limited by the slowest device." << endl;
    }

    // Free stream memory for each device
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numKernels; j++)
        {
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_x[j]));
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_y[j]));
        }
    }

    // Display metric values
    cout << endl << "Metrics for device #0:" << endl;
    cout << "Look at the sm__cycles_elapsed.max values for each test." << endl;
    cout << "This value represents the time spent on device to run the kernels in each case, and should be longest for the serial range, and roughly equal for the single and multi device concurrent ranges." << endl;
    // PrintMetricValues(deviceData[0].config.chipName, deviceData[0].counterDataImage, metricNames);
    print_measured_values(deviceData[0], metricNames);

    // Only display next device info if needed
    if (numDevices > 1)
    {
        cout << endl << "Metrics for the remaining devices only display the multi device async case and should all be similar to the first device's values if the device has similar performance characteristics." << endl;
        cout << "If devices have different performance characteristics, the runtime cycles calculation may vary by device." << endl;
    }
    for (int i = 1; i < numDevices; i++)
    {
        cout << endl << "Metrics for device #" << i << ":" << endl;
        // PrintMetricValues(deviceData[i].config.chipName, deviceData[i].counterDataImage, metricNames);
        print_measured_values(deviceData[i], metricNames);
    }
    PAPI_shutdown();
    return 0;
}
