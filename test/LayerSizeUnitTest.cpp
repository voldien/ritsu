#include "core/Shape.h"
#include "layers/Input.h"
#include <Ritsu.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LayerSizeTest : public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerSizeTest, InputSize) {
	auto [x, expected] = GetParam();

	Ritsu::Input input(x);

	EXPECT_EQ(input.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(InputSize, LayerSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16}),
														   Ritsu::Shape<uint32_t>({16}))));

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

TEST_P(LayerSizeTest, Values) {
	auto [x, expected] = GetParam();

	Ritsu::Dense dense(16);

	EXPECT_EQ(dense.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(DenseLayer, LayerSizeTest,
						 ::testing::Values(std::make_tuple(Ritsu::Shape<uint32_t>({16}),
														   Ritsu::Shape<uint32_t>({16}))));

// class LayerSizeTest : public ::testing::TestWithParam<std::tuple<Ritsu::Shape<uint32_t>, Ritsu::Shape<uint32_t>>> {};

TEST_P(LayerSizeTest, UpscaleLayerSize) {
	auto [x, expected] = GetParam();

	Ritsu::UpSampling2D<float> upscale(2, UpSampling2D<float>::Interpolation::NEAREST);

	EXPECT_EQ(upscale.getShape(), expected);
}

// INSTANTIATE_TEST_SUITE_P(DenseLayer, LayerSizeTest, ::testing::Values(std::make_tuple({16}, {16} )));
