#include "core/Shape.h"
#include <Ritsu.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class OptimizerTest : public ::testing::TestWithParam<std::tuple<float, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(OptimizerTest, Setup) {
	auto [x, y, expected] = GetParam();

	SGD sgd(0.02f);
}

TEST_P(OptimizerTest, UpgradeVariable) {
	auto [x, y, expected] = GetParam();

	SGD sgd(0.02f);

	Tensor<float> gradient;
	Tensor<float> variable;

	ASSERT_NO_THROW(sgd.update_step(gradient, variable));
}

INSTANTIATE_TEST_SUITE_P(Optimizer, OptimizerTest,
						 ::testing::Values(std::make_tuple(0.02, 1, Ritsu::Shape<uint32_t>({1}))));
