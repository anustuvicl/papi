/**
 * @file    lcuda_config.h
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#ifndef __LCUDA_CONFIG_H__
#define __LCUDA_CONFIG_H__

#include <cupti.h>

#define CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION  (13)
// #define CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION (xx)  // TODO: Set to last version when CUPTI events API removed

#if CUPTI_API_VERSION >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION
#   define API_PERFWORKS 1
#endif

// #   if CUPTI_API_VERSION <= CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION
#define API_EVENTS 1
// #   endif

#endif  // __LCUDA_CONFIG_H__
