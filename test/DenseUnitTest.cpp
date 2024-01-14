#include "Tensor.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class DenseTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {};

TEST_P(DenseTest, Setup) {
	auto [x, y, expected] = GetParam();

	Input a0({x, 1});
	Dense a1(y);

	Layer<float> &output = a0(a1);
	EXPECT_EQ(output.getShape(), expected);
}

TEST_P(DenseTest, ComputeShape) {
	auto [x, y, expected] = GetParam();

	Input a0({x, 1});
	Dense a1(y);

	Tensor a({x, 1});

	Layer<float> &output = a0(a1);
	Tensor result = a0(a);

	EXPECT_EQ(result.getShape(), expected);
	EXPECT_EQ(a1.getTrainableWeights()->getShape(), Ritsu::Shape<uint32_t>({x, y}));
}

// TODO:
TEST_P(DenseTest, ComputeResult) {
	auto [x, y, expected] = GetParam();

	Input a0({x, 1});
	Dense a1(y);

	Tensor a({x, 1});

	Layer<float> &output = a0(a1);
	Tensor result = a0(a);

	EXPECT_EQ(result.getShape(), expected);
}

TEST_P(DenseTest, ComputeDerivativeResult) {
	auto [x, y, expected] = GetParam();

	Input a0({x, 1});
	Dense a1(y);

	Tensor a({x, 1});

	Layer<float> &output = a0(a1);
	Tensor result = a0(a);

	EXPECT_EQ(result.getShape(), expected);
}

INSTANTIATE_TEST_SUITE_P(Dense, DenseTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16 * 32}))));