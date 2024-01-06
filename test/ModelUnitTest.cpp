#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class ModelTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(ModelTest, Setup) {
	auto [x, y, expected] = GetParam();

	// EXPECT_EQ(output.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Setup, ModelTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16 * 32}))));