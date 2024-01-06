#include "Activations.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class ActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double>> {};

TEST_P(ActivationFunctionTest, Relu) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(relu(x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationRelu, ActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
/*	*/

TEST_P(ActivationFunctionTest, Sigmoid) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeSigmoid(x), expected_activation);
}

INSTANTIATE_TEST_SUITE_P(ActivationSigmoid, ActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
/*	*/

TEST_P(ActivationFunctionTest, Linear) {
	auto [x, expected_activation] = GetParam();

	// EXPECT_FLOAT_EQ(computeLinear(x), expected_activation);
}

INSTANTIATE_TEST_SUITE_P(ActivationLinear, ActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
/*	*/

TEST_P(ActivationFunctionTest, Tahn) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeTanh(x), expected_activation);
}

TEST_P(ActivationFunctionTest, Softmax) {
	// auto [x, min, max, expected] = GetParam();
	// auto clampedValue = Math::clamp(x, min, max);
	//
	// EXPECT_FLOAT_EQ(clampedValue, expected);
}

TEST_P(ActivationFunctionTest, Swish) {
	// auto [x, min, max, expected] = GetParam();
	// auto clampedValue = Math::clamp(x, min, max);
	//
	// EXPECT_FLOAT_EQ(clampedValue, expected);
}

TEST_P(ActivationFunctionTest, LeakyRelu) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(leakyRelu(x, 0.2), expected_activation);
}
