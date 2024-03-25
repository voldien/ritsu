#include "Loss.h"
#include "Tensor.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LossTest : public ::testing::TestWithParam<std::tuple<Tensor<float>, Tensor<float>, Tensor<float>>> {};

TEST_P(LossTest, LossError) {
	auto [x, y, expected] = GetParam();

	Tensor<float> result;
	ASSERT_NO_THROW(Ritsu::loss_error(x, y, result));
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, MSE) {
	auto [x, y, expected] = GetParam();

	Tensor<float> result;
	ASSERT_NO_THROW(Ritsu::loss_mse(x, y, result));
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, MSA) {
	auto [x, y, expected] = GetParam();

	Tensor<float> result;
	ASSERT_NO_THROW(Ritsu::loss_msa(x, y, result));
	ASSERT_EQ(result, expected);
}

INSTANTIATE_TEST_SUITE_P(Loss, LossTest,
						 ::testing::Values(std::make_tuple(Tensor<float>::fromArray({0, 0, 0, 0}),
														   Tensor<float>::fromArray({0, 0, 0, 0}),
														   Tensor<float>::fromArray({0}))));

class LossBinaryCrossEntropyTest
	: public ::testing::TestWithParam<std::tuple<Tensor<float>, Tensor<float>, Tensor<float>>> {};

TEST_P(LossBinaryCrossEntropyTest, loss_binary_cross_entropy) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;
	// CategoricalCrossentropy crossEntrpy;
	// ASSERT_NO_THROW(Ritsu::loss_binary_cross_entropy(x, y, result));
	// ASSERT_EQ(result, expected);
}

INSTANTIATE_TEST_SUITE_P(Loss, LossBinaryCrossEntropyTest,
						 ::testing::Values(std::make_tuple(Tensor<float>::fromArray({0, 0, 0, 0}),
														   Tensor<float>::fromArray({0, 0, 0, 0}),
														   Tensor<float>::fromArray({0}))));
//
TEST_P(LossTest, loss_cross_catagorial_entropy) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;

	ASSERT_NO_THROW(Ritsu::loss_categorical_crossentropy(x, y, result));
	//ASSERT_EQ(result, expected);
}

TEST_P(LossTest, sparse_categorical_crossentropy) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;

	ASSERT_NO_THROW(Ritsu::sparse_categorical_crossentropy(x, y, result));
	//ASSERT_EQ(result, expected);
}
