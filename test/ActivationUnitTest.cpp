#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

class ReluTest : public ::testing::TestWithParam<std::tuple<double, double>> {};

TEST_P(ReluTest, ActivationFunction) {
	// auto [x, min, max, expected] = GetParam();
	// auto clampedValue = Math::clamp(x, min, max);
	//
	// EXPECT_FLOAT_EQ(clampedValue, expected);
}

INSTANTIATE_TEST_SUITE_P(Activation, ReluTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
