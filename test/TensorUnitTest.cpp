#include <Tensor.h>
#include <gtest/gtest.h>
using namespace Ritsu;

template <class T> class TensorType : public ::testing::Test {};
TYPED_TEST_SUITE_P(TensorType);

TYPED_TEST_P(TensorType, DefaultConstructor) {
	// Tensor<TypeParam> shape;
	ASSERT_NO_THROW(Tensor tensor({1, 1}, sizeof(TypeParam)));
}

TYPED_TEST_P(TensorType, SetGetValues) {
	Tensor tensor({1, 1}, sizeof(TypeParam));

	tensor.getValue<TypeParam>(0) = (TypeParam)0;

	ASSERT_EQ(tensor.getValue<TypeParam>(0), (TypeParam)0);
}

REGISTER_TYPED_TEST_SUITE_P(TensorType, DefaultConstructor, SetGetValues);

using TensorPrimitiveDataTypes = ::testing::Types<uint16_t, uint32_t, size_t, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Parameter, TensorType, TensorPrimitiveDataTypes);