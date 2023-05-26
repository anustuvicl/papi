/**
 * @file    lcuda_common.c
 * @author  Anustuv Pal
 *          anustuv@icl.utk.edu
 */

#include <string.h>
#include "papi_memory.h"

#include "lcuda_common.h"
#include "lcuda_htable.h"
#include "lcuda_debug.h"

#define ADDED_EVENTS_INITIAL_CAPACITY 64

event_list_t *initialize_dynamic_event_list_size(int size)
{
    event_list_t *evt_table = (event_list_t *) papi_malloc(sizeof(event_list_t));
    if (evt_table == NULL)
        goto fn_exit;
    evt_table->capacity = size;
    evt_table->count = 0;
    evt_table->evts = (event_rec_t *) papi_calloc (evt_table->capacity, sizeof(event_rec_t));
    if (evt_table->evts == NULL) {
        free_event_name_list(&evt_table);
        ERRDBG("Error allocating memory for dynamic event table.\n");
        goto fn_exit;
    }
    if (htable_init(&(evt_table->htable)) != HTABLE_SUCCESS) {
        free_event_name_list(&evt_table);
    }
fn_exit:
    return evt_table;
}

event_list_t *initialize_dynamic_event_list(void)
{
    return initialize_dynamic_event_list_size(ADDED_EVENTS_INITIAL_CAPACITY);
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
        if (HTABLE_SUCCESS != htable_insert(evt_table->htable, evt_table->evts[i].name, &(evt_table->evts[i])))
            return PAPI_ENOMEM;
    }
    return PAPI_OK;
}

int insert_event_record(event_list_t *evt_table, const char *evt_name, unsigned int evt_code, int evt_pos)
{
    int errno = PAPI_OK;

    // Allocate twice the space if running out
    if (evt_table->count >= evt_table->capacity) {
        errno = reallocate_array(evt_table);
        if (errno != PAPI_OK)
            goto fn_exit;
    }
    // Insert record in array
    strcpy(evt_table->evts[evt_table->count].name, evt_name);
    evt_table->evts[evt_table->count].desc[0] = '\0';
    evt_table->evts[evt_table->count].evt_code = evt_code;
    evt_table->evts[evt_table->count].evt_pos = evt_pos;
    // Insert entry in string hash table
    if (HTABLE_SUCCESS != htable_insert(evt_table->htable, evt_name, &(evt_table->evts[evt_table->count]))) {
        return PAPI_ENOMEM;
    }
    evt_table->count ++;
fn_exit:
    return errno;
}

int find_event_name(event_list_t *evt_table, const char *evt_name, event_rec_t **found_rec)
{
    int errno;

    event_rec_t *evt_rec = NULL;
    errno = htable_find(evt_table->htable, evt_name, (void **) &evt_rec);
    if (errno == HTABLE_SUCCESS) {
        *found_rec = evt_rec;
        return PAPI_OK;
    }
    return PAPI_ENOEVNT;
}

void free_event_name_list(event_list_t **pevt_table)
{
    event_list_t *evt_table = *pevt_table;
    if (evt_table == NULL)
        return;
    if (evt_table->evts) {
        papi_free(evt_table->evts);
        evt_table->evts = NULL;
    }
    if (evt_table->htable) {
        htable_shutdown(evt_table->htable);
        evt_table->htable = NULL;
    }
    papi_free(evt_table);
    *pevt_table = NULL;
}

int tokenize_event_name(const char *name, char *nv_name, int *gpuid)
{
    // Resolve the nvidia name and gpu number from PAPI event name
    int numchars;

    if (nv_name == NULL) {
        return PAPI_EINVAL;
    }
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
    int errno = PAPI_OK, gpu_id;
    long i;
    char nv_name[PAPI_2MAX_STR_LEN];
    gpu_occupancy_t acq_mask = 0;
    for (i = 0; i < evt_table->count; i++) {
        errno = tokenize_event_name(evt_table->evts[0].name, nv_name, &gpu_id);
        if (errno != PAPI_OK)
            goto fn_exit;
        acq_mask |= (1 << gpu_id);
    }
    *bitmask = acq_mask;
fn_exit:
    return errno;
}

int devmask_check_and_acquire(event_list_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int errno = _devmask_events_get(evt_table, &bitmask);
    if (errno != PAPI_OK)
        return errno;
    if (bitmask & global_gpu_bitmask) {
        return PAPI_ECNFLCT;
    }
    global_gpu_bitmask |= bitmask;
    return PAPI_OK;
}

int devmask_release(event_list_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int errno = _devmask_events_get(evt_table, &bitmask);
    if (errno != PAPI_OK)
        return errno;
    if ((bitmask & global_gpu_bitmask) != bitmask) {
        return PAPI_EMISC;
    }
    global_gpu_bitmask ^= bitmask;
    return PAPI_OK;
}

int search_files_in_path(const char *file_name, const char *search_path, char **file_paths)
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
