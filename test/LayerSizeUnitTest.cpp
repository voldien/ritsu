#include "layers/Dropout.h"
#include "layers/Input.h"
#include <Ritsu.h>
#include <array>
#include <cstddef>
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

	EXPECT_EQ(input.getShape(), expected);
}

/*	*/
TEST_P(LayerUniformShapeSizeTest, RescalingLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Rescaling rescalling(1);
	rescalling(input);
	rescalling.build(input.getShape());

	EXPECT_EQ(rescalling.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, ReluLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Relu relu;
	relu(input);
	relu.build(input.getShape());

	EXPECT_EQ(relu.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, LeakyReluLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::LeakyRelu leakyRelu(0.2f);
	leakyRelu(input);
	leakyRelu.build(input.getShape());

	EXPECT_EQ(leakyRelu.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SigmoidLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Sigmoid sigmoid;
	sigmoid(input);
	sigmoid.build(input.getShape());

	EXPECT_EQ(sigmoid.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SoftMaxLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::SoftMax softmax;
	softmax(input);
	softmax.build(input.getShape());

	EXPECT_EQ(softmax.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SwishLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Swish swish;
	swish(input);
	swish.build(input.getShape());

	EXPECT_EQ(swish.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, TahnLayerShapeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Tanh tanh;
	tanh(input);
	tanh.build(input.getShape());

	EXPECT_EQ(tanh.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, BatchNormalizeSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::BatchNormalization batchNormalize;
	batchNormalize(input);
	batchNormalize.build(input.getShape());

	EXPECT_EQ(batchNormalize.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, GaussianNoiseSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::GuassianNoise guassianNoise(0.2f, 0.0f);
	guassianNoise(input);
	guassianNoise.build(input.getShape());

	EXPECT_EQ(guassianNoise.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, RegularizationSize) {
	auto [expected] = GetParam();

	Ritsu::Input input(expected);
	Ritsu::Regularization regularization;
	regularization(input);
	regularization.build(input.getShape());

	EXPECT_EQ(regularization.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, Add) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);

	Ritsu::Add add;
	add(input0, input1);

	EXPECT_EQ(add.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Subtract) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);
	Ritsu::Subtract subtract;
	subtract(input0, input1);

	EXPECT_EQ(subtract.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Multiply) {
	auto [expected] = GetParam();
	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);
	Ritsu::Multiply<float> multiply;
	multiply(input0, input1);

	EXPECT_EQ(multiply.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Divide) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);
	Ritsu::Input input1(expected);

	Ritsu::Divide divide;
	divide(input0, input1);

	EXPECT_EQ(divide.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Cast) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);

	Ritsu::Cast<int, float> cast;
	cast(input0);
	cast.build(input0.getShape());

	EXPECT_EQ(cast.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Dropout) {
	auto [expected] = GetParam();

	Ritsu::Input input0(expected);

	Ritsu::Dropout dropout(0.2f);
	dropout(input0);
	dropout.build(input0.getShape());

	EXPECT_EQ(dropout.getShape(), expected);
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

	EXPECT_EQ(flatten.getShape(), expected);
}
/*	*/

INSTANTIATE_TEST_SUITE_P(
	Flatten, LayerFlattenShapeSizeTest,
	::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}), Ritsu::Shape<uint32_t>({768})),
					  std::make_tuple(Ritsu::Shape<uint32_t>({32, 32, 32, 3}), Ritsu::Shape<uint32_t>({98304}))));

class LayerDown2DScaleShapeSizeTest
	: public ::testing::TestWithParam<
		  std::tuple<std::array<uint32_t, 2>, Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerDown2DScaleShapeSizeTest, MaxPooling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::MaxPooling2D maxPooling2D(stride);
	maxPooling2D(input);
	maxPooling2D.build(input.getShape());

	EXPECT_EQ(maxPooling2D.getShape(), expected);
}
/*	*/

TEST_P(LayerDown2DScaleShapeSizeTest, MinPooling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::MinPooling2D minPooling2D(stride);
	minPooling2D(input);
	minPooling2D.build(input.getShape());

	EXPECT_EQ(minPooling2D.getShape(), expected);
}
/*	*/

TEST_P(LayerDown2DScaleShapeSizeTest, AveragePooling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::AveragePooling2D averagePooling2D(stride);
	averagePooling2D(input);
	averagePooling2D.build(input.getShape());

	EXPECT_EQ(averagePooling2D.getShape(), expected);
}
/*	*/

TEST_P(LayerDown2DScaleShapeSizeTest, Conv2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Conv2D conv2D(64, {2, 2}, stride, ConvPadding::Same);
	conv2D(input);
	conv2D.build(input.getShape());

	EXPECT_EQ(conv2D.getShape(), expected);
}
/*	*/

INSTANTIATE_TEST_SUITE_P(
	Math, LayerDown2DScaleShapeSizeTest,
	::testing::Values(std::make_tuple(std::array<uint32_t, 2>({2, 2}), Ritsu::Shape<uint32_t>({16, 16, 3}),
									  Ritsu::Shape<uint32_t>({8, 8, 3})),
					  std::make_tuple(std::array<uint32_t, 2>({2, 2}), Ritsu::Shape<uint32_t>({256, 256, 1}),
									  Ritsu::Shape<uint32_t>({128, 128, 1}))));

class LayerUp2DScaleShapeSizeTest
	: public ::testing::TestWithParam<
		  std::tuple<std::array<uint32_t, 2>, Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

/*	*/
TEST_P(LayerUp2DScaleShapeSizeTest, UpSampling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::UpSampling2D<float> upscale(2, UpSampling2D<float>::Interpolation::NEAREST);

	Ritsu::Input input(x);
	upscale(input);
	upscale.build(input.getShape());

	EXPECT_EQ(upscale.getShape(), expected);
}

TEST_P(LayerUp2DScaleShapeSizeTest, Conv2DTranspose) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Conv2DTranspose conv2DTranspose(64, {2, 2});
	conv2DTranspose(input);
	conv2DTranspose.build(input.getShape());

	EXPECT_EQ(conv2DTranspose.getShape(), expected);
}
INSTANTIATE_TEST_SUITE_P(
	Math, LayerUp2DScaleShapeSizeTest,
	::testing::Values(std::make_tuple(std::array<uint32_t, 2>({2, 2}), Ritsu::Shape<uint32_t>({16, 16, 3}),
									  Ritsu::Shape<uint32_t>({8, 8, 3})),
					  std::make_tuple(std::array<uint32_t, 2>({2, 2}), Ritsu::Shape<uint32_t>({256, 256, 1}),
									  Ritsu::Shape<uint32_t>({128, 128, 1}))));

class LayerConcatenateShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};
TEST_P(LayerConcatenateShapeSizeTest, Concatenate) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(x);

	Ritsu::Concatenate cat({}, "");
	cat(input0, input1);

	EXPECT_EQ(cat.getShape(), expected);
}

// TODO: add one more argument, from to, expected
class LayerReshapeShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};
TEST_P(LayerReshapeShapeSizeTest, Reshape) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);

	Reshape reshape(expected);
	reshape(input0);
	reshape.build(input0.getShape());

	EXPECT_EQ(reshape.getShape(), expected);
}
INSTANTIATE_TEST_SUITE_P(Reshape, LayerReshapeShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}),
														   Ritsu::Shape<uint32_t>({2, 8, 8, 3}))));

// TEST_P(LayerUniformShapeSizeTest, Values) {
//	auto [x, expected] = GetParam();
//
//	Ritsu::Dense dense(16);
//
//	EXPECT_EQ(dense.getShape(), expected);
//}

// INSTANTIATE_TEST_SUITE_P(DenseLayer, LayerUniformShapeSizeTest,
//						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16}),
//														   Ritsu::Shape<uint32_t>({16}))));

class NonUniformSizeTest : public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {

};
