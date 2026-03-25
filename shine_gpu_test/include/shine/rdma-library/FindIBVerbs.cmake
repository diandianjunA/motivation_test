# FindIBVerbs.cmake
# CMake module to find libibverbs

find_path(IBVerbs_INCLUDE_DIR
    NAMES infiniband/verbs.h
    PATHS /usr/include
)

find_library(IBVerbs_LIBRARY
    NAMES ibverbs
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IBVerbs
    REQUIRED_VARS
        IBVerbs_LIBRARY
        IBVerbs_INCLUDE_DIR
)

if(IBVerbs_FOUND)
    set(IBVERBS_LIBRARY ${IBVerbs_LIBRARY})
    set(IBVERBS_INCLUDE_DIR ${IBVerbs_INCLUDE_DIR})
endif()
