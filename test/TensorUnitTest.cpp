#include "core/Shape.h"
#include <Tensor.h>
#include <cstdint>
#include <gtest/gtest.h>
using namespace Ritsu;

template <class T> class TensorType : public ::testing::Test {};
TYPED_TEST_SUITE_P(TensorType);

TYPED_TEST_P(TensorType, DefaultConstructor) {
	ASSERT_NO_THROW(Tensor tensor({32, 32, 3}, sizeof(TypeParam)));
	ASSERT_NO_THROW(Tensor tensor(Shape<uint32_t>({32, 32, 3}), sizeof(TypeParam)));
}

TYPED_TEST_P(TensorType, AssignMove) {
	Tensor tensor({32, 32, 3}, sizeof(TypeParam));
	ASSERT_NO_THROW(Tensor b = std::move(tensor));
}

TYPED_TEST_P(TensorType, ElementCount) {
	Tensor tensor({32, 32, 3}, sizeof(TypeParam));
	ASSERT_EQ(tensor.getNrElements(), 32 * 32 * 3);

	Tensor tensorB({32, 32, 6}, sizeof(TypeParam));
	ASSERT_EQ(tensorB.getNrElements(), 32 * 32 * 6);
}

// TODO:
TYPED_TEST_P(TensorType, FromArray) {
	// Tensor<TypeParam> shape;
	ASSERT_NO_THROW(Tensor tensor({32, 32, 3}, sizeof(TypeParam)));
}

TYPED_TEST_P(TensorType, SetGetValues) {
	Tensor tensor({32, 32, 3}, sizeof(TypeParam));

	for (size_t i = 0; i < tensor.getNrElements(); i++) {
		const TypeParam randomValue = static_cast<TypeParam>(rand());
		tensor.getValue<TypeParam>(i) = (TypeParam)randomValue;
		ASSERT_EQ(tensor.getValue<TypeParam>(i), (TypeParam)randomValue);
	}
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

TYPED_TEST_P(TensorType, Flatten) {

	Tensor tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.getValue<TypeParam>(0) = (TypeParam)randomValue;

	auto &ref = tensor.flatten();

	ASSERT_EQ(tensor.getValue<TypeParam>(0), (TypeParam)randomValue);
	ASSERT_EQ(tensor.getShape(), Shape<unsigned int>({32 * 32 * 3}));
}

TYPED_TEST_P(TensorType, InnerProduct) {

	Tensor tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorA.assignInitValue(1);
	tensorB.assignInitValue(1);

	ASSERT_NEAR(Tensor::dot(tensorA, tensorB), (8 * 8 * 3) * 1, 0.0001f);
}

TYPED_TEST_P(TensorType, Append) {

	Tensor tensorA(Shape<uint32_t>({3}), sizeof(TypeParam));
	Tensor tensorB(Shape<uint32_t>({3}), sizeof(TypeParam));

	tensorB.assignInitValue(static_cast<TypeParam>(1));
	tensorA.assignInitValue(static_cast<TypeParam>(1));

	ASSERT_NO_THROW(tensorA.append(tensorB));

	ASSERT_EQ(tensorA.getShape(), Shape<uint32_t>({6}));

	for (size_t i = 0; i < tensorA.getNrElements(); i++) {
		EXPECT_EQ(tensorA.getValue<TypeParam>(i), 1);
	}
}

TYPED_TEST_P(TensorType, Cast) {

	Tensor tensorA(Shape<uint32_t>({3}), sizeof(TypeParam));
	const Tensor tensorB(Shape<uint32_t>({3}), sizeof(TypeParam));

	Tensor ref;
	ASSERT_NO_THROW(tensorA.cast<int16_t>());
	ASSERT_NO_THROW(ref = std::move(tensorB.cast<int16_t>()));

	ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(int16_t));
	ASSERT_EQ(ref.getDatSize(), ref.getNrElements() * sizeof(int16_t));
}

TYPED_TEST_P(TensorType, SubSet) {

	Tensor tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	Tensor subset = std::move(tensorA.getSubset(0, 1, Shape<uint32_t>({1})));

	// TODO:
	ASSERT_EQ(subset.getShape(), Shape<uint32_t>({8, 8, 3}));
}

TYPED_TEST_P(TensorType, MatrixMultiplication) {

	Tensor tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorB.assignInitValue(1);
	tensorA.assignInitValue(1);

	// TODO:
	// ASSERT_EQ(subset.getShape(), Shape<uint32_t>({8, 8, 3}));
}

TYPED_TEST_P(TensorType, Equal) {

	Tensor tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorB.assignInitValue(1);
	tensorA.assignInitValue(1);

	// TODO:
	ASSERT_EQ(tensorA, tensorB);
}

REGISTER_TYPED_TEST_SUITE_P(TensorType, DefaultConstructor, AssignMove, ElementCount, FromArray, SetGetValues, Log10,
							Mean, Flatten, InnerProduct, Append, Cast, SubSet, MatrixMultiplication, Equal);

using TensorPrimitiveDataTypes = ::testing::Types<uint16_t, uint32_t, size_t, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Parameter, TensorType, TensorPrimitiveDataTypes);

// resize
// Number of elements.
// Sub shape
// Axis dim