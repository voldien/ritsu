#include "RitsuUnitTest.h"
#include <Ritsu.h>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class OptimizerSetupTest : public ::testing::TestWithParam<std::tuple<float, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(OptimizerSetupTest, SetupSGD) {
	const auto [learning_rate, y, expected] = GetParam();

	SGD sgd(learning_rate);
	ASSERT_EQ(sgd.getLearningRate(), learning_rate);
	ASSERT_STREQ(sgd.getName().c_str(), "sgd");
}

TEST_P(OptimizerSetupTest, SetupAdam) {
	const auto [learning_rate, y, expected] = GetParam();

	Adam<float> adam(learning_rate, 0.02, 0.01f);
	ASSERT_EQ(adam.getLearningRate(), learning_rate);
	ASSERT_STREQ(adam.getName().c_str(), "adam");
}

TEST_P(OptimizerSetupTest, SetupAda) {
	const auto [learning_rate, y, expected] = GetParam();

	Ada<float> ada(learning_rate, 0.02);
	ASSERT_EQ(ada.getLearningRate(), learning_rate);
	ASSERT_STREQ(ada.getName().c_str(), "ada");
}

INSTANTIATE_TEST_SUITE_P(Optimizer, OptimizerSetupTest,
						 ::testing::Values(std::make_tuple(0.02, 1, Ritsu::Shape<uint32_t>({1}))));

class OptimizerGradientTest : public ::testing::TestWithParam<std::tuple<float, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(OptimizerSetupTest, SGDGradientVariable) {
	const auto [learning_rate, y, expected] = GetParam();

	SGD sgd_optimizer(learning_rate);
	ASSERT_EQ(sgd_optimizer.getLearningRate(), learning_rate);

	Tensor<float> gradient(expected);
	Tensor<float> variable(expected);

	ASSERT_NO_THROW(sgd_optimizer.update_step(gradient, variable));
}

TEST_P(OptimizerSetupTest, SGDApplyGradientVariable) {
	const auto [learning_rate, y, expected] = GetParam();

	SGD sgd_optimizer(learning_rate);
	ASSERT_EQ(sgd_optimizer.getLearningRate(), learning_rate);

	Tensor<float> gradient(expected);
	Tensor<float> variable(expected);

	ASSERT_NO_THROW(sgd_optimizer.apply_gradients(gradient, variable));
}

TEST_P(OptimizerSetupTest, AdamGradientVariable) {
	const auto [learning_rate, y, expected] = GetParam();

	Adam<float> adam_optimizer(learning_rate);
	ASSERT_EQ(adam_optimizer.getLearningRate(), learning_rate);

	Tensor<float> gradient(expected);
	Tensor<float> variable(expected);

	ASSERT_NO_THROW(adam_optimizer.update_step(gradient, variable));
}

INSTANTIATE_TEST_SUITE_P(Optimizer, OptimizerGradientTest,
						 ::testing::Values(std::make_tuple(0.02, 1, Ritsu::Shape<uint32_t>({1}))));
