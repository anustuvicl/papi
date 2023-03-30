#include <papi.h>
#include <stdio.h>

#define LOG(...) \
do { \
fprintf(stderr, "LOG MAIN: ln %d: ", __LINE__); \
fprintf(stderr, __VA_ARGS__); \
} while (0);

int main()
{
    int retval = PAPI_library_init( PAPI_VER_CURRENT );
    if( retval != PAPI_VER_CURRENT ) {
        fprintf( stderr, "Please recompile this test program. Installed PAPI has been updated.\n" );
        exit(-1);
    }

    int cid = PAPI_get_component_index("cuda_pw");
    LOG("cuda_pw component index = %d", cid);

    return 0;
}