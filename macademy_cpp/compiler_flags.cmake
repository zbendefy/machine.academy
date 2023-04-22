if (NOT DEFINED COMPILER_FLAGS_DEFAULT)
    # turn on C++17 features

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21")
        set(CMAKE_C_STANDARD 17)
    else()
        set(CMAKE_C_STANDARD 11)
    endif()

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        ## TODO:
        ## maybe-uninitialized should be useful
        set(COMPILER_FLAG_DISABLE_TO_STRICT_WARNINGS "--warn-no-maybe-uninitialized")
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(COMPILER_FLAG_ALL_WARNINGS "-Wall" "-Wextra" "-Wformat=2")
        set(COMPILER_FLAG_PEDANTIC_WARNINGS "-Wpedantic")
        set(COMPILER_FLAG_TREAT_WARNINGS_AS_ERRORS "-Werror")
        set(COMPILER_FLAG_GLM "-DGLM_ENABLE_EXPERIMENTAL" "-DGLM_FORCE_SWIZZLE" "-DGLM_FORCE_DEPTH_ZERO_TO_ONE")
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(COMPILER_FLAG_COVERAGE "--coverage")
            if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                set(COMPILER_FLAG_DEBUG "-D_DEBUG")
            endif()
        endif()
        set(COMPILER_FLAG_SUPPORT_SHARED "-fPIC")
        set(COMPILER_FLAG_OLD_STYLE_CAST "$<$<COMPILE_LANGUAGE:CXX>:-Wold-style-cast>")
        if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(COMPILER_FLAG_VECTOR_EXTENSIONS "") #NEON is enabled on arm64
        else()
            set(COMPILER_FLAG_VECTOR_EXTENSIONS "-mavx")
        endif()
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10.0 AND "${USE_TIME_TRACE}" STREQUAL "1")
        set(COMPILER_FLAG_TIME_TRACE "-ftime-trace")
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(COMPILER_FLAG_ALL_WARNINGS "/W3")
        set(COMPILER_FLAG_REORDER "/w25038")
        set(COMPILER_FLAG_TREAT_WARNINGS_AS_ERRORS "/WX")
        set(COMPILER_FLAG_MULTIPLE_PROCESSES "/MP")
        set(COMPILER_FLAG_UNREFERENCED_PARAMETER "/w34100")
        set(COMPILER_FLAG_GLM "/DGLM_ENABLE_EXPERIMENTAL" "/DGLM_FORCE_SWIZZLE" "/DGLM_FORCE_DEPTH_ZERO_TO_ONE")
        set(COMPILER_FLAG_SUPPRESS_WARNINGS "/wd4250")
        set(COMPILER_FLAG_ENABLE_BIGOBJ "/bigobj")
        set(COMPILER_FLAG_VECTOR_EXTENSIONS "/arch:AVX")
        set(COMPILER_FLAG_UNREFERENCED_PARAMETER "/w24100")                            # eggs variant not compatiable with this
        set(COMPILER_FLAG_UNREFERENCED_LOCAL_VARIABLE "/w24101")                       # unreferenced local variable
        set(COMPILER_FLAG_LOCAL_VARIABLE_INITED_BUT_NOT_REFERENCED "/w24189")          # local variable is initialized but not referenced
        set(COMPILER_FLAG_WIN32_LEAN_AND_MEAN "/DWIN32_LEAN_AND_MEAN=1")               # currently limited usage because VMA_USAGE wrongly declarated it
        set(COMPILER_FLAG_PARALLEL_LINKER "/cgthreads8")
        set(LINKER_FLAG_FASTLINK "/DEBUG:FASTLINK") # Generate Debug Information optimized for faster links

        set(COMPILER_FLAG_ITERATOR_BASE_CLASS_DEPRECATION_WARNING "/D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING")

        # False warnings in vs2017, see:
        # https://developercommunity.visualstudio.com/content/problem/500588/boost-asio-reports-stdallocator-is-deprecated-in-c.html
        # https://github.com/chriskohlhoff/asio/issues/290#issuecomment-371867040
        if (MSVC_TOOLSET_VERSION LESS "142")
            set(COMPILER_FLAG_RESULT_OF_DEPRECATION_WARNING "/D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING")
            set(COMPILER_FLAG_ALLOCATOR_VOID_DEPRECATION_WARNING "/D_SILENCE_CXX17_ALLOCATOR_VOID_DEPRECATION_WARNING")
        endif()
    endif()

    set(COMPILER_FLAGS_DEFAULT #${COMPILER_FLAG_ALL_WARNINGS}
                               ${COMPILER_FLAG_REORDER}
                               #${COMPILER_FLAG_TREAT_WARNINGS_AS_ERRORS}
                               ${COMPILER_FLAG_DISABLE_TO_STRICT_WARNINGS}
                               ${COMPILER_FLAG_MULTIPLE_PROCESSES}
                               ${COMPILER_FLAG_UNREFERENCED_PARAMETER}
                               ${COMPILER_FLAG_DEBUG}
                               ${COMPILER_FLAG_UNREFERENCED_PARAMETER}
                               ${COMPILER_FLAG_UNREFERENCED_LOCAL_VARIABLE}
                               ${COMPILER_FLAG_LOCAL_VARIABLE_INITED_BUT_NOT_REFERENCED}
                               ${COMPILER_FLAG_SUPPORT_SHARED}
                               ${COMPILER_FLAG_ENABLE_BIGOBJ}
                               ${COMPILER_FLAG_PARALLEL_LINKER}
                               ${COMPILER_FLAG_OLD_STYLE_CAST}
                               ${COMPILER_FLAG_TIME_TRACE}
                               ${COMPILER_FLAG_GLM}
                               ${COMPILER_FLAG_ITERATOR_BASE_CLASS_DEPRECATION_WARNING}
                               ${COMPILER_FLAG_RESULT_OF_DEPRECATION_WARNING}
                               ${COMPILER_FLAG_ALLOCATOR_VOID_DEPRECATION_WARNING}
                               ${COMPILER_FLAG_SUPPRESS_WARNINGS}
                               ${COMPILER_FLAG_VECTOR_EXTENSIONS})

    set(COMPILER_FLAGS_STRICT ${COMPILER_FLAGS_DEFAULT} ${COMPILER_FLAG_PEDANTIC_WARNINGS})

    if(WIN32)

        # https://stackoverflow.com/a/58020501/149111 suggests that /Z7 should be used; rather than placing the debug info into a .pdb
        # In order for /Zc:inline, which speeds up the build significantly, to work, we need to remove the /Ob0 parameter

        foreach(flag_var CMAKE_C_FLAGS_CHECKEDWITHDEBINFO CMAKE_CXX_FLAGS_CHECKEDWITHDEBINFO)

            if (${flag_var} MATCHES "/Zi")
                string(REGEX REPLACE "/Zi" "/Z7" ${flag_var} "${${flag_var}}")
            endif()

            if (${flag_var} MATCHES "/Ob0")
                string(REGEX REPLACE "/Ob0" "" ${flag_var} "${${flag_var}}")
            endif()

        endforeach()
    endif()

    # add compile options for the current directory and below
    #add_compile_options(${COMPILER_FLAGS_STRICT})

endif()
