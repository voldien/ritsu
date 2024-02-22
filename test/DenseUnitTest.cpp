#include "RitsuUnitTest.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class DenseOutShape : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseOutShape, Setup) {
	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({xUnit});
	Dense *dense;
	ASSERT_NO_THROW(dense = new Dense(denseUnit));

	Layer<float> *output = nullptr;
	ASSERT_NO_THROW(output = &(*dense)(input));

	ASSERT_EQ(output->getShape(), expected);
	ASSERT_STREQ(output->getName().c_str(), "dense");

	ASSERT_NO_THROW(delete dense);
}

TEST_P(DenseOutShape, SetupInvalidShapeThrow) {
	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({1, 1, xUnit});
	Dense dense = Dense(denseUnit);

	Layer<float> *output;
	ASSERT_THROW(output = &(dense(input)), RuntimeException);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseOutShape,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({32})),
										   std::make_tuple(32, 32, Ritsu::Shape<uint32_t>({32})),
										   std::make_tuple(1024, 10, Ritsu::Shape<uint32_t>({10}))));

class DenseParameterTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseParameterTest, WeightSize) {
	auto [xUnit, denseUnit, expectedWeightShape] = GetParam();

	Input input({xUnit});

	Dense *dense = nullptr;
	ASSERT_NO_THROW(dense = new Dense(denseUnit));

	Layer<float> &output = (*dense)(input);
	dense->build(dense->getShape());

	EXPECT_EQ(dense->getTrainableWeights()->getShape(), expectedWeightShape);
	EXPECT_EQ(dense->getVariables()->getShape(), Ritsu::Shape<uint32_t>({denseUnit}));

	ASSERT_NO_THROW(delete dense);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseParameterTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16, 32})),
										   std::make_tuple(32, 32, Ritsu::Shape<uint32_t>({32, 32})),
										   std::make_tuple(1024, 10, Ritsu::Shape<uint32_t>({1024, 10}))));

class DenseComputeTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseComputeTest, ResultShape) {

	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({xUnit});
	Dense dense(denseUnit);

	Tensor<float> inputData({xUnit});

	Layer<float> &output = dense(input); /*	Build the weight.	*/

	/*	*/
	// Tensor<float> result0 = dense(inputData);
	// Tensor<float> result1 = dense << inputData;

	// EXPECT_EQ(result0.getShape(), result1.getShape());

	/*	*/
	// EXPECT_EQ(result0.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseComputeTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16, 32})),
										   std::make_tuple(32, 32, Ritsu::Shape<uint32_t>({32, 32})),
										   std::make_tuple(1024, 10, Ritsu::Shape<uint32_t>({1024, 10}))));

class DenseTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseTest, ComputeDerivativeResult) {
	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({xUnit, 1});
	Dense dense(denseUnit);

	const Tensor<float> inputData({xUnit, 1});

	Layer<float> &output = dense(input);
	Tensor<float> result = dense.compute_derivative(inputData);

	EXPECT_EQ(result.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16 * 32}))));