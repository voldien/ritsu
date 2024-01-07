#include "layers/Dropout.h"
#include <Ritsu.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LayerUniformShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};
/*	*/
INSTANTIATE_TEST_SUITE_P(
	Input, LayerUniformShapeSizeTest,
	::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}), Ritsu::Shape<uint32_t>({16, 16, 3})),
					  std::make_tuple(Ritsu::Shape<uint32_t>({48, 48, 3}), Ritsu::Shape<uint32_t>({48, 48, 3}))));

TEST_P(LayerUniformShapeSizeTest, InputSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);

	EXPECT_EQ(input.getShape(), expected);
}

/*	*/
TEST_P(LayerUniformShapeSizeTest, RescalingLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Rescaling rescalling(1);
	rescalling(input);

	EXPECT_EQ(rescalling.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, ReluLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Relu relu;
	relu(input);

	EXPECT_EQ(relu.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, LeakyReluLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::LeakyRelu leakyRelu(0.2f);
	leakyRelu(input);

	EXPECT_EQ(leakyRelu.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SigmoidLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Sigmoid sigmoid;
	sigmoid(input);

	EXPECT_EQ(sigmoid.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SoftMaxLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::SoftMax softmax;
	softmax(input);

	EXPECT_EQ(softmax.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, SwishLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Swish swish;
	swish(input);

	EXPECT_EQ(swish.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, TanhLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Tahn tanh;
	tanh(input);

	EXPECT_EQ(tanh.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, BatchNormalizeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::BatchNormalization batchNormalize;
	batchNormalize(input);

	EXPECT_EQ(batchNormalize.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, GaussianNoiseSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::GuassianNoise guassianNoise(0.2f, 0.0f);
	guassianNoise(input);

	EXPECT_EQ(guassianNoise.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, RegularizationSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Regularization regularization;
	regularization(input);

	EXPECT_EQ(regularization.getShape(), expected);
}
/*	*/

TEST_P(LayerUniformShapeSizeTest, Add) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(x);

	Ritsu::Add add;

	EXPECT_EQ(add.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Subtract) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(x);
	Ritsu::Subtract subtract;

	EXPECT_EQ(subtract.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Multiply) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(x);
	Ritsu::Multiply<float> multiply;

	EXPECT_EQ(multiply.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Divide) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(x);

	Ritsu::Divide divide;

	EXPECT_EQ(divide.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Cast) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);

	Ritsu::Cast<float, int> cast;
	// cast(input0);

	EXPECT_EQ(cast.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Dropout) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);

	Ritsu::Dropout dropout(0.2f);
	dropout(input0);

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

	EXPECT_EQ(maxPooling2D.getShape(), expected);
}
/*	*/

TEST_P(LayerDown2DScaleShapeSizeTest, MinPooling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::MinPooling2D minPooling2D(stride);
	minPooling2D(input);

	EXPECT_EQ(minPooling2D.getShape(), expected);
}
/*	*/

TEST_P(LayerDown2DScaleShapeSizeTest, AveragePooling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::AveragePooling2D averagePooling2D(stride);
	averagePooling2D(input);

	EXPECT_EQ(averagePooling2D.getShape(), expected);
}
/*	*/

TEST_P(LayerDown2DScaleShapeSizeTest, Conv2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Conv2D conv2D(64, {2, 2}, stride, "");
	conv2D(input);

	EXPECT_EQ(conv2D.getShape(), expected);
}
/*	*/

INSTANTIATE_TEST_SUITE_P(
	Math, LayerDown2DScaleShapeSizeTest,
	::testing::Values(std::make_tuple(std::array<uint32_t, 2>({2, 2}), Ritsu::Shape<uint32_t>({16, 16, 3}),
									  Ritsu::Shape<uint32_t>({8, 8, 3})),
					  std::make_tuple(std::array<uint32_t, 2>({2, 2}), Ritsu::Shape<uint32_t>({256, 256, 1}),
									  Ritsu::Shape<uint32_t>({128, 128, 1}))));

/*	*/
TEST_P(LayerDown2DScaleShapeSizeTest, UpSampling2D) {
	auto [stride, x, expected] = GetParam();

	Ritsu::UpSampling2D<float> upscale(2, UpSampling2D<float>::Interpolation::NEAREST);

	EXPECT_EQ(upscale.getShape(), expected);
}

TEST_P(LayerDown2DScaleShapeSizeTest, Conv2DTranspose) {
	auto [stride, x, expected] = GetParam();

	Ritsu::Conv2DTranspose conv2DTranspose(64, {2, 2});

	EXPECT_EQ(conv2DTranspose.getShape(), expected);
}

class LayerConcatenateShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerConcatenateShapeSizeTest, Concatenate) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);
	Ritsu::Input input1(x);

	// Ritsu::Concatenate cat({&input0, &input1});

	// EXPECT_EQ(cat.getShape(), expected);
}

TEST_P(LayerConcatenateShapeSizeTest, Reshape) {
	auto [x, expected] = GetParam();

	Ritsu::Input input0(x);

	Reshape reshape(x);
	reshape(input0);

	EXPECT_EQ(reshape.getShape(), expected);
}

TEST_P(LayerUniformShapeSizeTest, Values) {
	auto [x, expected] = GetParam();

	Ritsu::Dense dense(16);

	EXPECT_EQ(dense.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(DenseLayer, LayerUniformShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16}),
														   Ritsu::Shape<uint32_t>({16}))));

class NonUniformSizeTest : public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {

};