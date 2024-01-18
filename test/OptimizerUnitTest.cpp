#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class OptimizerTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(OptimizerTest, Setup) {
	auto [x, y, expected] = GetParam();

	SGD sgd(0.02f);
}
