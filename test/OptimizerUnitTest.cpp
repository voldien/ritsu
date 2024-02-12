#include "RitsuUnitTest.h"
#include <Ritsu.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class OptimizerTest : public ::testing::TestWithParam<std::tuple<float, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(OptimizerTest, SetupSGD) {
	auto [learning_rate, y, expected] = GetParam();

	SGD sgd(learning_rate);
	ASSERT_EQ(sgd.getLearningRate(), learning_rate);
	ASSERT_STREQ(sgd.getName().c_str(), "sgd");
}

TEST_P(OptimizerTest, SetupAdam) {
	auto [learning_rate, y, expected] = GetParam();

	Adam<float> adam(learning_rate, 0.02, 0.01f);
	ASSERT_EQ(adam.getLearningRate(), learning_rate);
	ASSERT_STREQ(adam.getName().c_str(), "adam");
}

TEST_P(OptimizerTest, SetupAda) {
	auto [learning_rate, y, expected] = GetParam();

	Ada<float> ada(learning_rate, 0.02);
	ASSERT_EQ(ada.getLearningRate(), learning_rate);
	ASSERT_STREQ(ada.getName().c_str(), "ada");
}

TEST_P(OptimizerTest, UpgradeVariable) {
	auto [learning_rate, y, expected] = GetParam();

	SGD sgd(learning_rate);
	ASSERT_EQ(sgd.getLearningRate(), learning_rate);

	Tensor<float> gradient;
	Tensor<float> variable;

	ASSERT_NO_THROW(sgd.update_step(gradient, variable));
}

INSTANTIATE_TEST_SUITE_P(Optimizer, OptimizerTest,
						 ::testing::Values(std::make_tuple(0.02, 1, Ritsu::Shape<uint32_t>({1}))));
