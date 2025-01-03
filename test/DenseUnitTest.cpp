#include "RitsuUnitTest.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class DenseOutShape : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseOutShape, Setup) {
	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({xUnit});
	Dense *dense = nullptr;
	ASSERT_NO_THROW(dense = new Dense(denseUnit));

	Layer<float> *output = nullptr;
	ASSERT_NO_THROW(output = &(*dense)(input));
	output->build(input.getShape());

	ASSERT_EQ(output->getShape(), expected);
	ASSERT_STREQ(output->getName().c_str(), "dense");

	ASSERT_NO_THROW(delete dense);
}

TEST_P(DenseOutShape, SetupInvalidShapeThrow) {
	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({1, 1, xUnit});
	Dense dense = Dense(denseUnit);

	Layer<float> *output = &(dense(input));
	ASSERT_THROW(output->build(input.getShape()), RuntimeException);
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
	dense->build(input.getShape());

	// TODO:
	// ASSERT_EQ(dense->getTrainableWeights()->getShape(), expectedWeightShape);
	// ASSERT_EQ(dense->getVariables()->getShape(), Ritsu::Shape<uint32_t>({1, denseUnit}));

	ASSERT_NO_THROW(delete dense);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseParameterTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({32, 16})),
										   std::make_tuple(32, 32, Ritsu::Shape<uint32_t>({32, 32})),
										   std::make_tuple(1024, 10, Ritsu::Shape<uint32_t>({10, 1024}))));

class DenseComputeTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseComputeTest, ResultShape) {

	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({xUnit});
	Dense dense(denseUnit);

	Tensor<float> inputData({xUnit});

	Layer<float> &output = dense(input); /*	Build the weight.	*/
	output.build(dense.getInputs()[0]->getShape());

	/*	*/
	const Tensor<float> result0 = dense << inputData;

	/*	*/
	ASSERT_EQ(result0.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseComputeTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({32, 1})),
										   std::make_tuple(32, 32, Ritsu::Shape<uint32_t>({32, 1})),
										   std::make_tuple(1024, 10, Ritsu::Shape<uint32_t>({10, 1}))));

class DenseShapeTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseShapeTest, ComputeDerivativeResult) {
	auto [xUnit, denseUnit, expected] = GetParam();

	Input input({xUnit});
	Dense dense(denseUnit);

	const Tensor<float> inputData({1, denseUnit});

	Layer<float> &output = dense(input);
	output.build(dense.getInputs()[0]->getShape());

	Tensor<float> result;
	ASSERT_NO_THROW(result = dense.compute_derivative(inputData.transpose()));

	ASSERT_EQ(result.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseShapeTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16, 1})),
										   std::make_tuple(32, 32, Ritsu::Shape<uint32_t>({32, 1})),
										   std::make_tuple(1024, 10, Ritsu::Shape<uint32_t>({1024, 1}))));