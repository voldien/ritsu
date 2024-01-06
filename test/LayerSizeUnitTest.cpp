#include <Ritsu.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LayerUniformShapeSizeTest
	: public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerUniformShapeSizeTest, InputSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);

	EXPECT_EQ(input.getShape(), expected);
}

/*	*/
INSTANTIATE_TEST_SUITE_P(Input, LayerUniformShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}),
														   Ritsu::Shape<uint32_t>({16, 16, 3})),
										   std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 16, 3}),
														   Ritsu::Shape<uint32_t>({16, 16, 16, 3}))));
/*	*/
TEST_P(LayerUniformShapeSizeTest, RescalingLayerShapeSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);
	Ritsu::Rescaling rescalling(1);
	rescalling(input);

	EXPECT_EQ(rescalling.getShape(), expected);
}

// INSTANTIATE_TEST_SUITE_P(Rescaling, LayerUniformSizeTest,
//						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16}),
//														   Ritsu::Shape<uint32_t>({16}))));
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

INSTANTIATE_TEST_SUITE_P(Flatten, LayerFlattenShapeSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}),
														   Ritsu::Shape<uint32_t>({768}))));

// TEST_P(LayerSizeTest, Add) {
//	auto [x, expected] = GetParam();
//
//	Ritsu::Add add;
//
//	EXPECT_EQ(add.getShape(), expected);
//}
//
// TEST_P(LayerSizeTest, Subtract) {
//	auto [x, expected] = GetParam();
//
//	Ritsu::Subtract add;
//
//	EXPECT_EQ(add.getShape(), expected);
//}
//
// TEST_P(LayerSizeTest, AveragePooling2D) {
//	auto [x, expected] = GetParam();
//
//	Ritsu::AveragePooling2D add(2);
//
//	EXPECT_EQ(add.getShape(), expected);
//}

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

TEST_P(NonUniformSizeTest, Upscale) {
	auto [x, expected] = GetParam();

	Ritsu::UpSampling2D<float> upscale(2, UpSampling2D<float>::Interpolation::NEAREST);

	EXPECT_EQ(upscale.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(UpscaleLayerSize, NonUniformSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16, 16, 3}),
														   Ritsu::Shape<uint32_t>({32, 32, 3})))); /*	*/