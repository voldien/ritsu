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

class ReluDerivativeActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(ReluDerivativeActivationFunctionTest, Relu) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(reluDeriviate(x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(Relu, ReluDerivativeActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 1), std::make_tuple(3, 1), std::make_tuple(-1000, 0)));

class LeakyReluActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {};
TEST_P(LeakyReluActivationFunctionTest, LeakyRelu) {
	auto [alpha, x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(leakyRelu(alpha, x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(LeakyRelu, LeakyReluActivationFunctionTest,
						 ::testing::Values(std::make_tuple(0.2, 5, 5), std::make_tuple(0.2, 1, 1),
										   std::make_tuple(0.2, -10, -2)));

class LeakyReluDerivativeActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {
};
TEST_P(LeakyReluDerivativeActivationFunctionTest, LeakyRelu) {
	auto [alpha, x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(leakyReluDerivative(alpha, x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(LeakyRelu, LeakyReluDerivativeActivationFunctionTest,
						 ::testing::Values(std::make_tuple(0.2, 5, 1), std::make_tuple(0.2, 1, 1),
										   std::make_tuple(0.2, -1000, 0.2)));

/*	*/
class SigmoidActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(SigmoidActivationFunctionTest, Sigmoid) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeSigmoid(x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(Sigmoid, SigmoidActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 0.99330714907572), std::make_tuple(1, 0.7310585786300),
										   std::make_tuple(-1, 0.26894142136999512)));

/*	Derivative	*/
class SigmoidDerivativeActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(SigmoidDerivativeActivationFunctionTest, Sigmoid) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeSigmoidDerivate(x), expected_activation);
}

// TODO:
INSTANTIATE_TEST_SUITE_P(ActivationSigmoid, SigmoidDerivativeActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 0.006648056670790154),
										   std::make_tuple(1, 0.1966119332414818525374),
										   std::make_tuple(-1, 0.19661193324148185253742473358590)));

/*	*/
class LinearActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {};
TEST_P(LinearActivationFunctionTest, Linear) {
	auto [coff, x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeLinear(coff, x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationLinear, LinearActivationFunctionTest,
						 ::testing::Values(std::make_tuple(0.2, 5, 1), std::make_tuple(0.3, 1, 0.3),
										   std::make_tuple(0.5, -1000, -500)));

/*	Derivative	*/
class LinearDerivativeActivationFunctionTest : public LinearActivationFunctionTest {};
TEST_P(LinearDerivativeActivationFunctionTest, Linear) {
	auto [coff, x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeLinearDerivative(coff), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationLinear, LinearDerivativeActivationFunctionTest,
						 ::testing::Values(std::make_tuple(0.2, 5, 0.2), std::make_tuple(0.3, 1, 0.3),
										   std::make_tuple(0.5, -1000, 0.5)));

class ExpLinearActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {};
TEST_P(ExpLinearActivationFunctionTest, ExpLinear) {
	auto [coff, x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeExpLinear(coff, x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationExpLinear, ExpLinearActivationFunctionTest,
						 ::testing::Values(std::make_tuple(0.2, 5.0, 5.0), std::make_tuple(0.2, 1.0, 1),
										   std::make_tuple(0.2, -10, -0.19999092001)));
/*	Derivative	*/
class ExpLinearDerivativeActivationFunctionTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {
};
TEST_P(ExpLinearDerivativeActivationFunctionTest, ExpLinear) {
	auto [coff, x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeExpLinearDerivative(coff, x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationDerivativeExpLinear, ExpLinearDerivativeActivationFunctionTest,
						 ::testing::Values(std::make_tuple(0.2, 5, 1), std::make_tuple(0.2, 1, 1),
										   std::make_tuple(0.2, -5, 0.00134758939)));

/*	*/
class TanhActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(TanhActivationFunctionTest, Tahn) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeTanh(x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationTahn, TanhActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 0.9999092042625951312),
										   std::make_tuple(1, 0.76159415595576),
										   std::make_tuple(-2, -0.9640275800758)));

class TanhDerivativeActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(TanhDerivativeActivationFunctionTest, Tahn) {
	auto [x, expected_activation] = GetParam();

	EXPECT_FLOAT_EQ(computeTanhDerivate(x), expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationTahn, TanhDerivativeActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 0.000181583230943),
										   std::make_tuple(1, 0.419974341614026069394),
										   std::make_tuple(-2, 0.07065082485316446)));

class SwishActivationFunctionTest : public ActivationFunctionTest {};
TEST_P(SwishActivationFunctionTest, Swish) {
	auto [x, expected_activation] = GetParam();

	const auto clampedValue = computeSwish(x, 1.0);
	//
	EXPECT_FLOAT_EQ(clampedValue, expected_activation);
}
INSTANTIATE_TEST_SUITE_P(ActivationSwish, SwishActivationFunctionTest,
						 ::testing::Values(std::make_tuple(5, 4), std::make_tuple(1, 3), std::make_tuple(-1000, 3)));

class SoftMaxActivationFunctionTest : public ::testing::TestWithParam<std::tuple<Tensor<float>, Tensor<float>>> {};

TEST_P(SoftMaxActivationFunctionTest, Softmax) {
	auto [input, expected] = GetParam();

	const Tensor<float> result = softMax<float>(input);

	ASSERT_EQ(result.getShape(), expected.getShape());
	for (size_t i = 0; i < result.getNrElements(); i++) {
		ASSERT_NEAR(result.getValue(i), expected.getValue(i), 0.00001);
	}
}

INSTANTIATE_TEST_SUITE_P(
	ActivationSoftmax, SoftMaxActivationFunctionTest,
	::testing::Values(std::make_tuple(Tensor<float>::fromArray({5.0f, 5.0f, 5.0f, 5.0f, 5.0f}),
									  Tensor<float>::fromArray({0.2f, 0.2f, 0.2f, 0.2f, 0.2f})),
					  std::make_tuple(Tensor<float>::fromArray({10, 0, 0}), Tensor<float>::fromArray({1, 0, 0})),
					  std::make_tuple(Tensor<float>::fromArray({5, 5, 5}), Tensor<float>::fromArray({0.2, 0.2, 0.2}))));