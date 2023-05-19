#ifndef __DEBUG_COMP_H__
#define __DEBUG_COMP_H__

#include "papi.h"
#include "papi_internal.h"

// Control panel
// #define EXIT_ON_ERROR  // if enabled cuda/cupti error exits to system else not
// #define TIME_MEASUREMENT  // if enabled activate TICK/TOCK style time measurement

// Macro to either exit or continue depending on switch
#define EXIT_OR_NOT
#ifdef EXIT_ON_ERROR
#   undef EXIT_OR_NOT
#   define EXIT_OR_NOT exit(-1)
#endif

// ANSI colored logging facility for CUDA component
#define ANSIRED     "\033[31;1m"
#define ANSIBLUE    "\033[36;1m"
#define ANSIGREEN   "\033[32;1m"
#define ANSIYELLOW  "\033[33;1m"
#define ANSIMAGENTA "\033[35;1m"
#define ANSIEND     "\033[0m"

/* Function calls */
#define COMPDBG(format, args...) SUBDBG(ANSIBLUE "COMPDEBUG: " format ANSIEND, ## args);

/* General log in white */
#define LOGDBG(format, args...) SUBDBG(ANSIEND "LOG: " format ANSIEND, ## args);

/* Lock and unlock calls in magenta */
#define LOCKDBG(format, args...) SUBDBG(ANSIMAGENTA "LOCK: " format ANSIEND, ## args);

/* ERROR bold red */
#define ERRDBG(format, args...) SUBDBG(ANSIRED "ERROR: " format ANSIEND, ## args);

/* Log cuda driver and runtime calls */
#define LOGCUDACALL(format, args...) SUBDBG(ANSIGREEN "CUDACALL: " format ANSIEND, ## args);

/* Log cupti and perfworks calls */
#define LOGCUPTICALL(format, args...) SUBDBG(ANSIYELLOW "CUPTICALL: " format ANSIEND, ## args);

#define STOPWATCH
#define TICK
#define TOCK
#define TIMEDBG(...)
#ifdef TIME_MEASUREMENT
#    include <time.h>
#    undef STOPWATCH
#    undef TICK
#    undef TOCK
#    undef TIMEDBG
#    define STOPWATCH clock_t _tick, _tock, _tdiff;
#    define TICK _tick = clock();
#    define TOCK _tock = clock(); _tdiff = _tock - _tick;
#    define TIMEDBG(...) SUBDBG("TIMEDEBUG: " __VA_ARGS__ "%f s.\n", (double) _tdiff / CLOCKS_PER_SEC);
#endif  // TIME_MEASUREMENT

#endif  // __DEBUG_COMP_H__