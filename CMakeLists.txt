CMAKE_MINIMUM_REQUIRED( VERSION 3.9...3.20 )
PROJECT(ritsu-deep-learning CXX)

# Project uses C++17
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

# Ouput directory for both library and executables.
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "" FORCE)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE INTERNAL "" FORCE)

OPTION(RITSU_BUILD_WITH_TEST "Enable Testing." ON)

OPTION(BUILD_SHARED_LIBS "Build package with shared libraries." ON)
IF(NOT BUILD_SHARED_LIBS)
	SET(CMAKE_EXE_LINKER_FLAGS "-static")
	SET(LINK_SEARCH_START_STATIC TRUE)
ELSE()
	MESSAGE(STATUS "Shared libraries enabled")
ENDIF(NOT BUILD_SHARED_LIBS)

#
OPTION(RITSU_WITH_MEM_JEMALLOC "Enable malloc replacement (http://www.canonware.com/jemalloc)" ON)
MARK_AS_ADVANCED(RITSU_WITH_MEM_JEMALLOC)

IF(RITSU_WITH_MEM_JEMALLOC)
	FIND_PACKAGE(JeMalloc)
	IF(NOT JeMalloc_FOUND)
		FIND_PACKAGE(PkgConfig REQUIRED)
		pkg_check_modules (JEMALLOC jemalloc)

		pkg_search_module(JEMALLOC REQUIRED jemalloc)
	ENDIF()
ENDIF()

######################################
#  All cmake build options.
######################################
OPTION(RITSU_BUILD_WITH_ASAN "Enable AddressSanitizer." OFF )
OPTION(RITSU_BUILD_WITH_UBSAN "Enable Undefined Behavior sanitizer." OFF )
OPTION(RITSU_BUILD_WITH_TSAN "Enable Thread sanitizer" OFF)

OPTION(RITSU_BUILD_WITH_COVERAGE "Enable Code coverage report analysis." OFF)
OPTION(RITSU_BUILD_WITH_PROFILER "Enable Code Profiling." OFF)



# Time profiling

# ASAN

#Vecotorize

# Core library
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/ritsu)

# Example Projects
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/example)

# Test
IF(RITSU_BUILD_WITH_TEST)
	ENABLE_TESTING()
	ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/test)
ENDIF()