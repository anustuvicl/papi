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
        return PAPI_ENOMEM;
    }
    htable_init(&(evt_table->htable));
    return PAPI_OK;
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

    char *getdevstr = strstr(name, token);
    if (getdevstr == NULL) {
        ERRDBG("Event name does not contain device number.\n");
        return PAPI_EINVAL;
    }
    getdevstr += tok_len;
    *gpuid = atoi(getdevstr);
    numchars = strlen(name) - strlen(getdevstr) - tok_len;
    strncpy(nv_name, name, numchars);
    nv_name[numchars] = '\0';

    return PAPI_OK;
}
