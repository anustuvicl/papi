#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi_memory.h"

#include "common.h"
#include "debug_comp.h"
#include "htable.h"

#define ADDED_EVENTS_INITIAL_CAPACITY 64

int initialize_dynamic_event_list_size(event_list_t * evt_table, int size)
{
    evt_table->capacity = size;
    evt_table->count = 0;
    evt_table->evts = (event_rec_t *) papi_calloc (evt_table->capacity, sizeof(event_rec_t));
    if (evt_table->evts == NULL) {
        ERRDBG("Error allocating memory for dynamic event table.\n");
        goto fn_fail;
    }
    if (htable_init(&(evt_table->htable)) != HTABLE_SUCCESS) {
        goto fn_fail;
    }
    return PAPI_OK;
fn_fail:
    return PAPI_ENOMEM;
}

int initialize_dynamic_event_list(event_list_t * evt_table)
{
    return initialize_dynamic_event_list_size(evt_table, ADDED_EVENTS_INITIAL_CAPACITY);
}

static int reallocate_array(event_list_t *evt_table)
{
    evt_table->capacity *= 2;
    evt_table->evts = (event_rec_t *) papi_realloc(evt_table->evts, evt_table->capacity * sizeof(event_rec_t));
    if (evt_table == NULL) {
        ERRDBG("Failed to expand event_table array.\n");
        return PAPI_ENOMEM;
    }
    // Rehash all the table entries
    unsigned int i;
    for (i=0; i<evt_table->count; i++) {
        htable_insert(evt_table->htable, evt_table->evts[i].name, &(evt_table->evts[i]));
    }
    return PAPI_OK;
}

int insert_event_record(event_list_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos)
{
    int res;

    // Allocate twice the space if running out
    if (evt_table->count >= evt_table->capacity) {
        res = reallocate_array(evt_table);
        if (res != PAPI_OK)
            goto fn_exit;
    }
    // Insert record in array
    strcpy(evt_table->evts[evt_table->count].name, evt_name);
    evt_table->evts[evt_table->count].desc[0] = '\0';
    evt_table->evts[evt_table->count].evt_code = evt_code;
    evt_table->evts[evt_table->count].evt_pos = evt_pos;
    // Insert entry in string hash table
    htable_insert(evt_table->htable, evt_name, &(evt_table->evts[evt_table->count]));
    evt_table->count ++;
fn_exit:
    return res;
}

int find_event_name(event_list_t *evt_table, const char *evt_name, event_rec_t **found_rec)
{
    int res;

    event_rec_t *evt_rec = NULL;
    res = htable_find(evt_table->htable, evt_name, (void **) &evt_rec);
    if (res == HTABLE_SUCCESS) {
        *found_rec = evt_rec;
        return PAPI_OK;
    }
    return PAPI_ENOEVNT;
}

void free_event_name_list(event_list_t *evt_table)
{
    papi_free(evt_table->evts);
    htable_shutdown(evt_table->htable);
    evt_table->evts = NULL;
}

int tokenize_event_name(const char * name, char * nv_name, int * gpuid)
{
    // Resolve the nvidia name and gpu number from PAPI event name
    int numchars;

    const char token[] = ":device=";
    const int tok_len = 8;
    char *rest;

    char *getdevstr = strstr(name, token);
    if (getdevstr == NULL) {
        ERRDBG("Event name does not contain device number.\n");
        return PAPI_EINVAL;
    }
    getdevstr += tok_len;
    *gpuid = strtol(getdevstr, &rest, 10);
    numchars = strlen(name) - strlen(getdevstr) - tok_len;
    memcpy(nv_name, name, numchars);
    nv_name[numchars] = '\0';

    return PAPI_OK;
}

// Functions based on bitmasking to detect gpu exclusivity
static gpu_occupancy_t global_gpu_bitmask;

static int _devmask_events_get(event_list_t *evt_table, gpu_occupancy_t *bitmask)
{
    int res, gpu_id;
    long i;
    char nv_name[PAPI_2MAX_STR_LEN];
    gpu_occupancy_t acq_mask = 0;
    for (i = 0; i < evt_table->count; i++) {
        res = tokenize_event_name(evt_table->evts[0].name, nv_name, &gpu_id);
        if (res != PAPI_OK)
            goto fn_exit;
        acq_mask |= (1 << gpu_id);
    }
    *bitmask = acq_mask;
fn_exit:
    return res;
}

int devmask_check_and_acquire(event_list_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int res = _devmask_events_get(evt_table, &bitmask);
    if (res != PAPI_OK)
        return res;
    if (bitmask & global_gpu_bitmask) {
        return PAPI_ECNFLCT;
    }
    global_gpu_bitmask |= bitmask;
    return PAPI_OK;
}

int devmask_release(event_list_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int res = _devmask_events_get(evt_table, &bitmask);
    if (res != PAPI_OK)
        return res;
    if ((bitmask & global_gpu_bitmask) != bitmask) {
        return PAPI_EMISC;
    }
    global_gpu_bitmask ^= bitmask;
    return PAPI_OK;
}

int search_files_in_path(const char* file_name, const char* search_path, char** file_paths)
{
    char path[PATH_MAX];
    char command[PATH_MAX];
    snprintf(command, PATH_MAX, "find %s -name %s", search_path, file_name);

    FILE *fp;
    fp = popen(command, "r");
    if (fp == NULL) {
        ERRDBG("Failed to run system command find using popen.\n");
        return -1;
    }

    int count = 0;
    while (fgets(path, PATH_MAX, fp) != NULL) {
        path[strcspn(path, "\n")] = 0;
        file_paths[count] = strdup(path);
        count++;
        if (count >= MAX_FILES) {
            break;
        }
    }

    pclose(fp);
    if (count == 0) {
        ERRDBG("%s not found in path PAPI_CUDA_ROOT.\n", file_name);
    }
    return count;
}
