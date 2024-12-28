#include "layers/Dropout.h"
#include "layers/Input.h"
#include <Ritsu.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LayerUniformShapeSizeTest : public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>>> {};
/*	*/
INSTANTIATE_TEST_SUITE_P(Input, LayerUniformShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3})),
										   std::make_tuple(Ritsu::Shape<uint32_t>({48, 48, 3})),
										   std::make_tuple(Ritsu::Shape<uint32_t>({48, 48, 48, 3}))));

TEST_P(LayerUniformShapeSizeTest, InputSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);

	ASSERT_EQ(input.getShape(), expected);
}

/*	*/
TEST_P(LayerUniformShapeSizeTest, RescalingLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Rescaling rescalling(1);
	rescalling(input);
	rescalling.build(input.getShape());

	ASSERT_EQ(rescalling.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, ReluLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Relu relu;
	relu(input);
	relu.build(input.getShape());

	ASSERT_EQ(relu.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, LeakyReluLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::LeakyRelu leakyRelu(0.2f);
	leakyRelu(input);
	leakyRelu.build(input.getShape());

	ASSERT_EQ(leakyRelu.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SigmoidLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Sigmoid sigmoid;
	sigmoid(input);
	sigmoid.build(input.getShape());

	ASSERT_EQ(sigmoid.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SoftMaxLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::SoftMax softmax;
	softmax(input);
	softmax.build(input.getShape());

	ASSERT_EQ(softmax.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SwishLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Swish swish(0.2f);
	swish(input);
	swish.build(input.getShape());

	ASSERT_EQ(swish.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, TahnLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Tanh tanh;
	tanh(input);
	tanh.build(input.getShape());

	ASSERT_EQ(tanh.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, BatchNormalizeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::BatchNormalization batchNormalize;
	batchNormalize(input);
	batchNormalize.build(input.getShape());

	ASSERT_EQ(batchNormalize.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, GaussianNoiseSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::GuassianNoise guassianNoise(0.2f, 0.0f);
	guassianNoise(input);
	guassianNoise.build(input.getShape());

	ASSERT_EQ(guassianNoise.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, RegularizationSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Regularization regularization;
	regularization(input);
	regularization.build(input.getShape());

	ASSERT_EQ(regularization.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, Add) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);

	Ritsu::Add add;
	add(input0, input1);

	ASSERT_EQ(add.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Subtract) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);
	Ritsu::Subtract subtract;
	subtract(input0, input1);

	ASSERT_EQ(subtract.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Multiply) {
	auto [expected] = GetParam();
	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);
	Ritsu::Multiply<float> multiply;
	multiply(input0, input1);

	ASSERT_EQ(multiply.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Divide) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);

	Ritsu::Divide divide;
	divide(input0, input1);

	ASSERT_EQ(divide.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Cast) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);

	Ritsu::Cast<int, float> cast;
	cast(input0);
	cast.build(input0.getShape());

	ASSERT_EQ(cast.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Dropout) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);

	Ritsu::Dropout dropout(0.2f);
	dropout(input0);
	dropout.build(input0.getShape());

	ASSERT_EQ(dropout.getShape(), expected);
}

/*	------------------------------*/
class LayerFlattenShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerFlattenShapeSizeTest, FlattenSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Flatten flatten;
	flatten(input);
	flatten.build(input.getShape());

	ASSERT_EQ(flatten.getShape(), expected);
}
/*	*/

INSTANTIATE_TEST_SUITE_P(
	Flatten, LayerFlattenShapeSizeTest,
	::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}), Ritsu::Shape<uint32_t>({768})),
					  std::make_tuple(Ritsu::Shape<uint32_t>({32, 32, 32, 3}), Ritsu::Shape<uint32_t>({98304}))));

class LayerConcatenateShapeSizeTest
	: public ::testing::TestWithParam<
		  std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerConcatenateShapeSizeTest, Concatenate) {
	auto [x, y, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(y);

	Ritsu::Concatenate cat({}, "concatenate");
	cat(input0, input1);

	ASSERT_EQ(cat.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Math, LayerConcatenateShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({32, 32, 1}),
														   Ritsu::Shape<uint32_t>({32, 32, 2}),
														   Ritsu::Shape<uint32_t>({32, 32, 3}))));

// TODO: add one more argument, from to, expected
class LayerReshapeShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};
TEST_P(LayerReshapeShapeSizeTest, Reshape) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);

	Reshape reshape(expected);
	reshape(input0);
	reshape.build(input0.getShape());

	ASSERT_EQ(reshape.getShape(), expected);
}
INSTANTIATE_TEST_SUITE_P(Reshape, LayerReshapeShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}),
														   Ritsu::Shape<uint32_t>({2, 8, 8, 3}))));
