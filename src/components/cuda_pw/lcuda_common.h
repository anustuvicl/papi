#ifndef __LCUDA_COMMON_H__
#define __LCUDA_COMMON_H__

#include <papi.h>

typedef struct eventname_id_s {
    char name[PAPI_2MAX_STR_LEN];
    char desc[PAPI_2MAX_STR_LEN];
    int gpu_id;
    unsigned int evt_code;
    unsigned int evt_pos;  // index of added event
    int num_dep;
    double value;
    void *info;  // API specific details
} event_rec_t;

typedef struct event_name_list_s {
    event_rec_t *evts;
    unsigned int count;
    unsigned int capacity;
    void *htable;
} event_list_t;

// These functions form a simple API to handle dynamic list of strings
int initialize_dynamic_event_list(event_list_t *evt_table);
int initialize_dynamic_event_list_size(event_list_t * evt_table, int size);
int insert_event_record(event_list_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos);
int find_event_name(event_list_t *evt_table, const char *evt_name, event_rec_t **found_rec);
void free_event_name_list(event_list_t *evt_table);
int tokenize_event_name(const char * name, char * nv_name, int * gpuid);

// Functions to track the occupancy of gpu counters in event sets
typedef int64_t gpu_occupancy_t;

int devmask_check_and_acquire(event_list_t *evt_table);
int devmask_release(event_list_t *evt_table);

// Utility to locate a file in a given path
#define MAX_FILES 100
int search_files_in_path(const char* file_name, const char* search_path, char** file_paths);

#endif  // __LCUDA_COMMON_H__
