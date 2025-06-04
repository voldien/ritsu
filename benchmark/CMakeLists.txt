LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
INCLUDE(benchmark)


# Source files.
FILE(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
FILE(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

ADD_EXECUTABLE(ritsu-benchmark-test ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(ritsu-benchmark-test ritsu benchmark)
TARGET_INCLUDE_DIRECTORIES(ritsu-benchmark-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ )
ADD_DEPENDENCIES(ritsu-benchmark-test ritsu)

INSTALL(TARGETS ritsu-benchmark-test DESTINATION bin)

IF (TARGET ritsu-rocm)

	ADD_EXECUTABLE(ritsu-benchmark-test-rocm ${SOURCE_FILES} ${HEADER_FILES})
	TARGET_LINK_LIBRARIES(ritsu-benchmark-test-rocm  ritsu-rocm benchmark)
	TARGET_INCLUDE_DIRECTORIES(ritsu-benchmark-test-rocm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ )
	ADD_DEPENDENCIES(ritsu-benchmark-test-rocm ritsu-rocm)

	INSTALL(TARGETS ritsu-benchmark-test-rocm DESTINATION bin)
	 
ENDIF()


ADD_EXECUTABLE(ritsu-benchmark-test-nomp ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(ritsu-benchmark-test-nomp ritsu-no-opm pthread benchmark )
TARGET_INCLUDE_DIRECTORIES(ritsu-benchmark-test-nomp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ )
ADD_DEPENDENCIES(ritsu-benchmark-test-nomp ritsu)

INSTALL(TARGETS ritsu-benchmark-test-nomp DESTINATION bin)