#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class DenseTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseTest, Values) {
	auto [x, y, expected] = GetParam();

	Dense a0(x);
	Dense a1(y);
    
	Layer<float> &output = a0(a1);

	EXPECT_EQ(output.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Math, DenseTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16 * 32}))));