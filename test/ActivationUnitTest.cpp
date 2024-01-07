#include "Activations.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class ActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double>> {};
class ReluActivationFunctionTest : public ActivationFunctionTest {};

TEST_P(ReluActivationFunctionTest, Relu) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(relu(x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationRelu, ReluActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 5), std::make_tuple(3, 3), std::make_tuple(-1000, 0)));
/*	*/

class SigmoidActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(SigmoidActivationFunctionTest, Sigmoid) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeSigmoid(x), expected_activation);
}

INSTANTIATE_TEST_SUITE_P(ActivationSigmoid, SigmoidActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
/*	*/

class LinearActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(LinearActivationFunctionTest, Linear) {
	auto [x, expected_activation] = GetParam();

	// EXPECT_FLOAT_EQ(computeLinear(x), expected_activation);
}

INSTANTIATE_TEST_SUITE_P(ActivationLinear, LinearActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
/*	*/

class TanhActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(TanhActivationFunctionTest, Tahn) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeTanh(x), expected_activation);
}

INSTANTIATE_TEST_SUITE_P(ActivationTahn, TanhActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));

class SoftMaxActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(SoftMaxActivationFunctionTest, Softmax) {
	// auto [x, min, max, expected] = GetParam();
	// auto clampedValue = Math::clamp(x, min, max);
	//
	// EXPECT_FLOAT_EQ(clampedValue, expected);
}

class SwishActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(SwishActivationFunctionTest, Swish) {
	// auto [x, min, max, expected] = GetParam();
	// auto clampedValue = Math::clamp(x, min, max);
	//
	// EXPECT_FLOAT_EQ(clampedValue, expected);
}

class LeakyReluActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(LeakyReluActivationFunctionTest, LeakyRelu) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(leakyRelu(x, 0.2), expected_activation);
}

//TODO:
class ExpLinearActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(ExpLinearActivationFunctionTest, ExpLinear) {
	auto [x, expected_activation] = GetParam();

	//EXPECT_FLOAT_EQ(leakyRelu(x, 0.2), expected_activation);
}

INSTANTIATE_TEST_SUITE_P(ActivationExpLinear, ExpLinearActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));
