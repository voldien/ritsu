#include "Tensor.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LossTest : public ::testing::TestWithParam<std::tuple<Tensor, Tensor, Tensor>> {};

TEST_P(LossTest, MSE) {
	auto [x, y, expected] = GetParam();

	Tensor result;
	ASSERT_NO_THROW(Ritsu::loss_mse(x, y, result));
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, MSA) {
	auto [x, y, expected] = GetParam();

	Tensor result;
	Ritsu::loss_msa(x, y, result);
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, loss_binary_cross_entropy) {
	auto [x, y, expected] = GetParam();
	Tensor result;
	Ritsu::loss_binary_cross_entropy(x, y, result);
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, loss_cross_catagorial_entropy) {
	auto [x, y, expected] = GetParam();
	Tensor result;
	Ritsu::loss_cross_catagorial_entropy(x, y, result);
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, sparse_categorical_crossentropy) {
	auto [x, y, expected] = GetParam();
	Tensor result;
	Ritsu::sparse_categorical_crossentropy(x, y, result);
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, SSIM) {
	auto [x, y, expected] = GetParam();
	Tensor result;
	Ritsu::loss_ssim(x, y, result);
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, PSNR) {
	auto [x, y, expected] = GetParam();
	Tensor result;
	//  Ritsu::loss_mse(x, y, result);
	ASSERT_EQ(result, expected);
}

INSTANTIATE_TEST_SUITE_P(Loss, LossTest,
						 ::testing::Values(std::make_tuple(Tensor::fromArray({0, 1, 0}), Tensor::fromArray({1, 1, 0}),
														   Tensor::fromArray({0, 1, 0}))));