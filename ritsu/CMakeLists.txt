# Define all file lists.
FILE(GLOB_RECURSE SOURCE_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
FILE(GLOB_RECURSE HEADER_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

FIND_PACKAGE(OpenMP REQUIRED)

#TODO: add one with and without openmp.
#TODO: add so
ADD_LIBRARY(ritsu ${SOURCE_FILES} ${HEADER_FILES})

TARGET_COMPILE_FEATURES(ritsu PUBLIC cxx_constexpr cxx_alias_templates cxx_raw_string_literals
	cxx_variadic_templates cxx_uniform_initialization cxx_right_angle_brackets cxx_nullptr
	cxx_generic_lambdas cxx_override cxx_noexcept cxx_aggregate_default_initializers)

TARGET_LINK_LIBRARIES(ritsu PRIVATE m pthread PUBLIC OpenMP::OpenMP_CXX)

#TARGET_COMPILE_OPTIONS(ritsu PUBLIC -fopenmp -offload-arch=gfx1100)

TARGET_INCLUDE_DIRECTORIES(ritsu PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

INSTALL(TARGETS ritsu DESTINATION lib)
