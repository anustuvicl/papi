#ifndef __DEBUG_COMP_H__
#define __DEBUG_COMP_H__

// Logging facility

// Control panel
#define COMP_DEBUG
#define LOG_GENERAL
#define LOG_COMP_CALLS
#define LOG_CUDA_CALLS
#define LOG_THREAD_INFO
#define LOG_LOCK_CALLS
#define TIME_MEASUREMENT
#define EXIT_ON_ERROR  // if defined cuda/cupti error exits to system else not

// Macro to either exit or continue depending on switch
#define EXIT_OR_NOT
#ifdef EXIT_ON_ERROR
#   undef EXIT_OR_NOT
#   define EXIT_OR_NOT exit(-1)
#endif

// Disabling declarations
#define _LOGHDR(name)
#define ANSIRED
#define ANSIBLUE
#define ANSIGREEN
#define ANSIYELLOW
#define ANSIMAGENTA
#define ANSIEND

#define THR_INFO
#define COMPDBG(...)
#define LOGDBG(...)
#define LOCKDBG(...)
#define ERRDBG(...)
#define LOGCUDACALL(...)
#define LOGCUPTICALL(...)

#define STOPWATCH
#define TICK
#define TOCK
#define TIMEDBG(...)

#ifdef COMP_DEBUG  // Enable logging

#   include <stdio.h>

#   undef _LOGHDR
#   undef ANSIRED
#   undef ANSIBLUE
#   undef ANSIGREEN
#   undef ANSIYELLOW
#   undef ANSIMAGENTA
#   undef ANSIEND

#   define _LOGHDR(name) fprintf(stderr, name ": %s: %s: %d: ", __FILE__, __func__, __LINE__);
#   define ANSIRED fprintf(stderr, "\033[31;1m");
#   define ANSIBLUE fprintf(stderr, "\033[36;1m");
#   define ANSIGREEN fprintf(stderr, "\033[32;1m");
#   define ANSIYELLOW fprintf(stderr, "\033[33;1m");
#   define ANSIMAGENTA fprintf(stderr, "\033[35;1m");
#   define ANSIEND fprintf(stderr, "\033[0m");

#   ifdef LOG_THREAD_INFO
#       undef THR_INFO
#       include "papi_internal.h"
#       define THR_INFO fprintf(stderr, "THR %lu: ", (_papi_hwi_thread_id_fn != NULL) ? _papi_hwi_thread_id_fn() : 0);
#   endif  // LOG_THREAD_INFO

#   ifdef LOG_COMP_CALLS
/* Function calls */
#       undef COMPDBG
#       define COMPDBG(...) \
        do { \
            ANSIBLUE \
            _LOGHDR("COMPDEBUG");  \
            THR_INFO;  \
            fprintf(stderr, __VA_ARGS__);  \
            ANSIEND  \
        } while (0);
#   endif  // LOG_COMP_CALLS

#   ifdef LOG_GENERAL
/* General log in white */
#       undef LOGDBG
#       define LOGDBG(...) \
        do { \
            _LOGHDR("LOG");  \
            THR_INFO;  \
            fprintf(stderr, __VA_ARGS__);  \
        } while (0);
#   endif  // LOG_GENERAL

#   ifdef LOG_LOCK_CALLS
/* Lock and unlock calls in magenta */
#       undef LOCKDBG
#       define LOCKDBG(...) \
        do { \
            ANSIMAGENTA;  \
            _LOGHDR("LOCK");  \
            THR_INFO;  \
            fprintf(stderr, __VA_ARGS__);  \
            ANSIEND;  \
        } while (0);
#   endif  // LOG_LOCK_CALLS

/* ERROR bold red */
#   undef ERRDBG
#   define ERRDBG(...) \
    do { \
        ANSIRED \
        _LOGHDR("ERROR");  \
        THR_INFO;  \
        fprintf(stderr, __VA_ARGS__);  \
        ANSIEND \
    } while (0);

#   ifdef LOG_CUDA_CALLS
#       undef LOGCUDACALL
#       undef LOGCUPTICALL

/* Log cuda driver and runtime calls */
#       define LOGCUDACALL(...) \
        do { \
            ANSIGREEN  \
            _LOGHDR("CUDACALL");  \
            THR_INFO;  \
            fprintf(stderr, __VA_ARGS__);  \
            ANSIEND  \
        } while (0);

/* Log cupti and perfworks calls */
#       define LOGCUPTICALL(...) \
        do { \
            ANSIYELLOW;  \
            _LOGHDR("CUPTICALL");  \
            THR_INFO;  \
            fprintf(stderr, __VA_ARGS__);  \
            ANSIEND  \
        } while (0);

#   endif  // LOG_CUDA_CALLS

#   ifdef TIME_MEASUREMENT
#       include <time.h>
#       undef STOPWATCH
#       undef TICK
#       undef TOCK
#       undef TIMEDBG
#       define STOPWATCH clock_t _tick, _tock, _tdiff;
#       define TICK _tick = clock();
#       define TOCK _tock = clock(); _tdiff = _tock - _tick;
#       define TIMEDBG(...) fprintf(stderr, "TIMEDEBUG: " __VA_ARGS__ "%f s.\n", (double) _tdiff / CLOCKS_PER_SEC);
#   endif

#endif  // DEBUG_COMP
#endif  // __DEBUG_COMP_H__