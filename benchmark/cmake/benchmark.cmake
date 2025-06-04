INCLUDE(FetchContent)

IF(NOT TARGET benchmark)


	FetchContent_Declare(benchmark_source
		GIT_REPOSITORY https://github.com/google/benchmark
		GIT_TAG v1.9.4
	)

	FetchContent_GetProperties(benchmark_source)

	IF(NOT benchmark_source_POPULATED)
		FetchContent_Populate(benchmark_source)

        OPTION(BENCHMARK_ENABLE_GTEST_TESTS OFF)
        OPTION(BENCHMARK_USE_BUNDLED_GTEST OFF)
        OPTION( BENCHMARK_ENABLE_TESTING OFF)
		ADD_SUBDIRECTORY(${benchmark_source_SOURCE_DIR} ${benchmark_source_BINARY_DIR} EXCLUDE_FROM_ALL)
	ELSE()
		MESSAGE( WARNING "Could not find benchmark source code")
	ENDIF()
ENDIF()

