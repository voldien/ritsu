#include <Tensor.h>
#include <gtest/gtest.h>
using namespace Ritsu;

template <class T> class TensorType : public ::testing::Test {};
TYPED_TEST_SUITE_P(TensorType);

TYPED_TEST_P(TensorType, DefaultConstructor) { ASSERT_NO_THROW(Tensor tensor({32, 32, 3}, sizeof(TypeParam))); }

// TODO:
TYPED_TEST_P(TensorType, FromArray) {
	// Tensor<TypeParam> shape;
	ASSERT_NO_THROW(Tensor tensor({32, 32, 3}, sizeof(TypeParam)));
}

TYPED_TEST_P(TensorType, SetGetValues) {
	Tensor tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.getValue<TypeParam>(0) = (TypeParam)randomValue;

	ASSERT_EQ(tensor.getValue<TypeParam>(0), (TypeParam)randomValue);
}

TYPED_TEST_P(TensorType, Log10) {
	Tensor tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.getValue<TypeParam>(0) = (TypeParam)randomValue;

	Tensor::log10(tensor);

	ASSERT_EQ(tensor.getValue<TypeParam>(0), (TypeParam)randomValue);
}

TYPED_TEST_P(TensorType, Mean) {
	Tensor tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.getValue<TypeParam>(0) = (TypeParam)randomValue;

	Tensor::mean<float>(tensor);
	Tensor::abs(tensor);

	ASSERT_EQ(tensor.getValue<TypeParam>(0), (TypeParam)randomValue);
}

REGISTER_TYPED_TEST_SUITE_P(TensorType, DefaultConstructor, FromArray, SetGetValues, Log10, Mean);

using TensorPrimitiveDataTypes = ::testing::Types<uint16_t, uint32_t, size_t, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Parameter, TensorType, TensorPrimitiveDataTypes);

// flatten
// resize
// append
// Number of elements.
// Sub shape
// Axis dim
// compute index