# Define all file lists.
FILE(GLOB_RECURSE SOURCE_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
FILE(GLOB_RECURSE HEADER_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CUDAToolkit QUIET)
FIND_PACKAGE(HIP QUIET)
FIND_PACKAGE(IntelDPCPP QUIET)

#########################
# CPU Ritsu 
#########################
ADD_LIBRARY(ritsu ${SOURCE_FILES} ${HEADER_FILES})
TARGET_COMPILE_FEATURES(ritsu PUBLIC cxx_constexpr cxx_alias_templates cxx_raw_string_literals
	cxx_variadic_templates cxx_uniform_initialization cxx_right_angle_brackets cxx_nullptr
	cxx_generic_lambdas cxx_override cxx_noexcept cxx_aggregate_default_initializers)
TARGET_LINK_LIBRARIES(ritsu PRIVATE m pthread PUBLIC OpenMP::OpenMP_CXX)
IF(RITSU_WITH_MEM_JEMALLOC)
	TARGET_LINK_LIBRARIES(ritsu PRIVATE ${JEMALLOC_LIBRARIES})
ENDIF()
TARGET_COMPILE_OPTIONS(ritsu PUBLIC -openmp-mapping)

TARGET_INCLUDE_DIRECTORIES(ritsu PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

INSTALL(TARGETS ritsu DESTINATION lib)

###################################
# No OpenMP Version (Single Core)
###################################
ADD_LIBRARY(ritsu-no-opm ${SOURCE_FILES} ${HEADER_FILES})
TARGET_COMPILE_FEATURES(ritsu-no-opm PUBLIC cxx_constexpr cxx_alias_templates cxx_raw_string_literals
	cxx_variadic_templates cxx_uniform_initialization cxx_right_angle_brackets cxx_nullptr
	cxx_generic_lambdas cxx_override cxx_noexcept cxx_aggregate_default_initializers)

TARGET_LINK_LIBRARIES(ritsu-no-opm PRIVATE m)
IF(RITSU_WITH_MEM_JEMALLOC)
	TARGET_LINK_LIBRARIES(ritsu-no-opm PRIVATE ${JEMALLOC_LIBRARIES})
ENDIF()

TARGET_INCLUDE_DIRECTORIES(ritsu-no-opm PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)


##########################
# OneAPI Ritsu version.
##########################

IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND IntelDPCPP_FOUND)
	MESSAGE(STATUS "OneAPI found:")
	ADD_LIBRARY(ritsu-oneapi ${SOURCE_FILES} ${HEADER_FILES})
	TARGET_COMPILE_FEATURES(ritsu-rocm PUBLIC cxx_constexpr cxx_alias_templates cxx_raw_string_literals
		cxx_variadic_templates cxx_uniform_initialization cxx_right_angle_brackets cxx_nullptr
		cxx_generic_lambdas cxx_override cxx_noexcept cxx_aggregate_default_initializers)

	TARGET_LINK_LIBRARIES(ritsu-oneapi PRIVATE m pthread PUBLIC OpenMP::OpenMP_CXX)

	TARGET_INCLUDE_DIRECTORIES(ritsu-oneapi PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)

	#TODO: add argument for arch.
	TARGET_COMPILE_OPTIONS(ritsu-oneapi PUBLIC -target x86_64-pc-linux-gnu -fopenmp -qopenmp -fopenmp-targets=spir64 matmul_offload.c)

ENDIF()

# Only supported on clang
##########################
# ROCM Ritsu version.
##########################
IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND HIP_FOUND)
	MESSAGE(STATUS "HIP found:")
	ADD_LIBRARY(ritsu-rocm ${SOURCE_FILES} ${HEADER_FILES})
	TARGET_COMPILE_FEATURES(ritsu-rocm PUBLIC cxx_constexpr cxx_alias_templates cxx_raw_string_literals
		cxx_variadic_templates cxx_uniform_initialization cxx_right_angle_brackets cxx_nullptr
		cxx_generic_lambdas cxx_override cxx_noexcept cxx_aggregate_default_initializers)

	TARGET_LINK_LIBRARIES(ritsu-rocm PRIVATE m pthread PUBLIC OpenMP::OpenMP_CXX)


	TARGET_INCLUDE_DIRECTORIES(ritsu-rocm PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)

	#TODO: add argument for arch.
	TARGET_COMPILE_OPTIONS(ritsu-rocm PUBLIC -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -offload-arch=gfx1100)

ENDIF()

##########################
# CUDA Ritsu version.
##########################
IF(CUDAToolkit_FOUND AND RITSU_CUDA)
	MESSAGE(STATUS "CUDA found:")
	#CMAKE_CUDA_COMPILER

	ADD_LIBRARY(ritsu-nv ${SOURCE_FILES} ${HEADER_FILES})
	
	TARGET_COMPILE_FEATURES(ritsu-nv PUBLIC cxx_constexpr cxx_alias_templates cxx_raw_string_literals
		cxx_variadic_templates cxx_uniform_initialization cxx_right_angle_brackets cxx_nullptr
		cxx_generic_lambdas cxx_override cxx_noexcept cxx_aggregate_default_initializers)
	TARGET_LINK_LIBRARIES(ritsu-nv PRIVATE m pthread PUBLIC OpenMP::OpenMP_CXX)

	TARGET_COMPILE_OPTIONS(ritsu-nv PRIVATE –mp=gpu -gpu=cc80 -Xcompiler -fgomp)

	TARGET_INCLUDE_DIRECTORIES(ritsu-nv PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)

ENDIF()
