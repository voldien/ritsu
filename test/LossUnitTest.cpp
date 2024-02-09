#include "Tensor.h"
#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class LossTest : public ::testing::TestWithParam<std::tuple<Tensor<float>, Tensor<float>, Tensor<float>>> {};
//template <class T> class TensorType : public ::testing::Test {};
//TYPED_TEST_SUITE_P(TensorType);

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

TEST_P(LossTest, loss_binary_cross_entropy) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;

	ASSERT_NO_THROW(Ritsu::loss_binary_cross_entropy(x, y, result));
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, loss_cross_catagorial_entropy) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;

	ASSERT_NO_THROW(Ritsu::loss_cross_catagorial_entropy(x, y, result));
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, sparse_categorical_crossentropy) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;

	ASSERT_NO_THROW(Ritsu::sparse_categorical_crossentropy(x, y, result));
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, SSIM) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;

	ASSERT_NO_THROW(Ritsu::loss_ssim(x, y, result));
	// ASSERT_EQ(result.getShape(), expected.)
	ASSERT_EQ(result, expected);
}

TEST_P(LossTest, PSNR) {
	auto [x, y, expected] = GetParam();
	Tensor<float> result;
	//  Ritsu::loss_mse(x, y, result);
	ASSERT_EQ(result, expected);
}

INSTANTIATE_TEST_SUITE_P(Loss, LossTest,
						 ::testing::Values(std::make_tuple(Tensor<float>::fromArray({0, 0, 0, 0}),
														   Tensor<float>::fromArray({0, 0, 0, 0}),
														   Tensor<float>::fromArray({0, 0, 0, 0}))));

// Loss Object


//REGISTER_TYPED_TEST_SUITE_P(TensorType, DefaultConstructor, DefaultType, AssignMove, DataSize, Addition, Subtract,
//							MultiplyFactor, ElementCount, FromArray, SetGetValues, Log10, Mean, Flatten, InnerProduct,
//							Append, Reduce, Reshape, Cast, SubSet, MatrixMultiplication, Equal, NotEqual);
//
//using TensorPrimitiveDataTypes = ::testing::Types<int16_t, uint16_t, int32_t, uint32_t, long, size_t, float, double>;
//INSTANTIATE_TYPED_TEST_SUITE_P(Parameter, TensorType, TensorPrimitiveDataTypes);
