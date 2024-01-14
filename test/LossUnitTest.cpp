#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

class LossTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(LossTest, MSE) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

TEST_P(LossTest, MSA) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

TEST_P(LossTest, loss_binary_cross_entropy) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

TEST_P(LossTest, loss_cross_catagorial_entropy) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

TEST_P(LossTest, sparse_categorical_crossentropy) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

TEST_P(LossTest, SSIM) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

TEST_P(LossTest, PSNR) {
	auto [x, y, expected] = GetParam();

	//  Ritsu::loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result)
}

INSTANTIATE_TEST_SUITE_P(Model, LossTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16 * 32}))));