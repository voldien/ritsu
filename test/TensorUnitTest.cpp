#include "core/Shape.h"
#include <Tensor.h>
#include <cstdint>
#include <gtest/gtest.h>
using namespace Ritsu;

template <class T> class TensorType : public ::testing::Test {};
TYPED_TEST_SUITE_P(TensorType);

TYPED_TEST_P(TensorType, DefaultConstructor) {
	ASSERT_NO_THROW(Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam)));
	ASSERT_NO_THROW(Tensor<TypeParam> tensor(Shape<uint32_t>({32, 32, 3}), sizeof(TypeParam)));
}

TYPED_TEST_P(TensorType, DefaultType) {
	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
	ASSERT_EQ(tensor.getDType(), typeid(TypeParam));
}

TYPED_TEST_P(TensorType, AssignMove) {
	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
	void *data = tensor.template getRawData<void>();
	ASSERT_NO_THROW(Tensor<TypeParam> moved = tensor);

	Tensor<TypeParam> moved;
	ASSERT_NO_THROW(moved = std::move(tensor));
	ASSERT_EQ(data, moved.template getRawData<void>());
	ASSERT_EQ(nullptr, tensor.template getRawData<void>());
}

TYPED_TEST_P(TensorType, Equal) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorB.assignInitValue(1);
	tensorA.assignInitValue(1);

	ASSERT_EQ(tensorA, tensorA);
	ASSERT_EQ(tensorA, tensorB);
}

TYPED_TEST_P(TensorType, NotEqual) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	const Tensor<TypeParam> tensorC(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorB.assignInitValue(1);
	tensorA.assignInitValue(0);

	ASSERT_NE(tensorA, tensorB);
}

TYPED_TEST_P(TensorType, DataSize) {
	const Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

	ASSERT_EQ(tensor.getDatSize(), tensor.getNrElements() * sizeof(TypeParam));
}

TYPED_TEST_P(TensorType, Addition) {
	Tensor<TypeParam> tensorA(Shape<uint32_t>({10}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({10}), sizeof(TypeParam));

	tensorA.assignInitValue(-1);
	tensorB.assignInitValue(1);

	const Tensor<TypeParam> result = tensorA + tensorB;

	ASSERT_EQ(result, Tensor<TypeParam>::zero(result.getShape()));
}

TYPED_TEST_P(TensorType, Subtract) {
	Tensor<TypeParam> tensorA(Shape<uint32_t>({10}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({10}), sizeof(TypeParam));

	tensorA.assignInitValue(1);
	tensorB.assignInitValue(1);

	const Tensor<TypeParam> result = tensorA - tensorB;
	ASSERT_EQ(result, Tensor<TypeParam>::zero(result.getShape()));
}

TYPED_TEST_P(TensorType, MultiplyFactor) {

	Tensor<TypeParam> expected(Shape<uint32_t>({10}), sizeof(TypeParam));
	const TypeParam value = rand() % 100;
	expected.assignInitValue(value);

	Tensor<TypeParam> tensorA(Shape<uint32_t>({10}), sizeof(TypeParam));
	tensorA.assignInitValue(1);

	const Tensor<TypeParam> result = tensorA * value;
	ASSERT_EQ(result, expected);
}

TYPED_TEST_P(TensorType, MatrixMultiplication) {

	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::identityMatrix(Shape<uint32_t>({2, 2}));
		const Tensor<TypeParam> tensorB = Tensor<TypeParam>::identityMatrix(Shape<uint32_t>({2, 2}));

		const Tensor<TypeParam> result0 = std::move(tensorA % tensorB);
		const Tensor<TypeParam> result1 = std::move(Tensor<TypeParam>::matrixMultiply(tensorA, tensorB));
		ASSERT_EQ(result0, result1);

		// verify the shape.
		ASSERT_EQ(result0.getShape(), Shape<uint32_t>({2, 2}));
		ASSERT_EQ(result0.getShape(), result1.getShape());

		for (unsigned int i = 0; i < result0.getShape().getAxisDimensions(0); i++) {
			ASSERT_EQ(result0.getValue({i, i}), 1);
		}
	}

	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::identityMatrix(Shape<uint32_t>({2, 2}));
		const Tensor<TypeParam> tensorB = Tensor<TypeParam>::identityMatrix(Shape<uint32_t>({2, 2}));

		const Tensor<TypeParam> result0 = tensorA % tensorB;
		const Tensor<TypeParam> result1 = Tensor<TypeParam>::matrixMultiply(tensorA, tensorB);
		ASSERT_EQ(result0, result1);

		// verify the shape.
		ASSERT_EQ(result0.getShape(), Shape<uint32_t>({2, 2}));
		ASSERT_EQ(result0.getShape(), result1.getShape());

		for (unsigned int i = 0; i < result0.getShape().getAxisDimensions(0); i++) {
			ASSERT_EQ(result0.getValue({i, i}), 1);
		}
	}

	/*	*/
	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::zero(Shape<uint32_t>({2, 2}));
		const Tensor<TypeParam> tensorB = Tensor<TypeParam>::zero(Shape<uint32_t>({2, 1}));

		const Tensor<TypeParam> result0 = tensorA % tensorB;
		const Tensor<TypeParam> result1 = Tensor<TypeParam>::matrixMultiply(tensorA, tensorB);
		ASSERT_EQ(result0, result1);

		// TODO: verify the shape.
		ASSERT_EQ(result0.getShape(), Shape<uint32_t>({2, 1}));
		ASSERT_EQ(result0.getShape(), result1.getShape());

		for (size_t i = 0; i < tensorA.getNrElements(); i++) {
			ASSERT_EQ(tensorA.template getValue<TypeParam>(i), static_cast<TypeParam>(0));
		}
	}

	/*	*/
	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::zero(Shape<uint32_t>({2, 4}));
		const Tensor<TypeParam> tensorB = Tensor<TypeParam>::zero(Shape<uint32_t>({4, 2}));

		const Tensor<TypeParam> result0 = tensorA % tensorB;
		const Tensor<TypeParam> result1 = Tensor<TypeParam>::matrixMultiply(tensorA, tensorB);
		ASSERT_EQ(result0, result1);

		// TODO: verify the shape.
		ASSERT_EQ(result0.getShape(), Shape<uint32_t>({2, 2}));
		ASSERT_EQ(result0.getShape(), result1.getShape());

		for (size_t i = 0; i < tensorA.getNrElements(); i++) {
			ASSERT_EQ(tensorA.template getValue<TypeParam>(i), static_cast<TypeParam>(0));
		}
	}
}

TYPED_TEST_P(TensorType, ElementCount) {

	const Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
	ASSERT_EQ(tensor.getNrElements(), 32 * 32 * 3);

	const Tensor<TypeParam> tensorB({32, 32, 6}, sizeof(TypeParam));
	ASSERT_EQ(tensorB.getNrElements(), 32 * 32 * 6);
}

TYPED_TEST_P(TensorType, FromArray) {

	{
		/*	*/
		ASSERT_NO_THROW(Tensor<TypeParam>::fromArray({1, 1, 1, 1, 1}));
	}

	{
		const size_t value = rand() % 100;
		const Tensor<TypeParam> fromArray =
			std::move(Tensor<TypeParam>::fromArray({value, value, value, value, value}));
		/*	*/
		ASSERT_EQ(fromArray.getNrElements(), 5);
		ASSERT_EQ(fromArray.getShape(), Shape<typename Tensor<TypeParam>::IndexType>({5}));

		for (size_t i = 0; i < fromArray.getNrElements(); i++) {
			ASSERT_EQ(fromArray.getValue(i), value);
		}
	}
}

TYPED_TEST_P(TensorType, SetGetValues) {
	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

	for (size_t i = 0; i < tensor.getNrElements(); i++) {
		const TypeParam randomValue = static_cast<TypeParam>(rand());
		tensor.template getValue<TypeParam>(i) = (TypeParam)randomValue;
		ASSERT_EQ(tensor.template getValue<TypeParam>(i), (TypeParam)randomValue);
	}
}

TYPED_TEST_P(TensorType, Log10) {

	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.template getValue<TypeParam>(0) = (TypeParam)randomValue;

	Tensor<TypeParam>::log10(tensor);

	ASSERT_EQ(tensor.template getValue<TypeParam>(0), (TypeParam)randomValue);
}

TYPED_TEST_P(TensorType, Mean) {

	{
		Tensor<TypeParam> tensor({32, 10}, sizeof(TypeParam));

		TypeParam value;
		ASSERT_NO_THROW(value = tensor.mean());
	}

	{
		Tensor<TypeParam> tensor({32, 10}, sizeof(TypeParam));

		ASSERT_NO_THROW(Tensor<TypeParam>::template mean<TypeParam>(tensor, -1));

		const Tensor<TypeParam> result = Tensor<TypeParam>::template mean<TypeParam>(tensor, -1);
		ASSERT_EQ(result.getShape(), Shape<typename Tensor<TypeParam>::IndexType>({10}));
	}

	{
		Tensor<TypeParam> tensor({32, 10, 10}, sizeof(TypeParam));

		ASSERT_NO_THROW(Tensor<TypeParam>::template mean<TypeParam>(tensor, -1));

		const Tensor<TypeParam> result = Tensor<TypeParam>::template mean<TypeParam>(tensor, -1);
		ASSERT_EQ(result.getShape(), Shape<typename Tensor<TypeParam>::IndexType>({10}));
	}
	
	/*	*/
	{
		// Tensor<TypeParam> tensor({2, 2, 3}, sizeof(TypeParam));
		//
		// const TypeParam randomValue = static_cast<TypeParam>(rand());
		// tensor.template getValue<TypeParam>(0) = (TypeParam)randomValue;
		//
		// const Tensor<TypeParam> result = Tensor<TypeParam>::mean(tensor, -1);
		//
		// Tensor<TypeParam>::template mean<float>(tensor);
		//
		// Tensor<TypeParam>::abs(tensor);
		//
		// ASSERT_EQ(tensor.template getValue<TypeParam>(0), (TypeParam)randomValue);
	}
}

TYPED_TEST_P(TensorType, Flatten) {

	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.template getValue<TypeParam>(0) = (TypeParam)randomValue;

	auto &ref = tensor.flatten();

	ASSERT_EQ(tensor.template getValue<TypeParam>(0), (TypeParam)randomValue);
	ASSERT_EQ(tensor.getShape(), Shape<unsigned int>({32 * 32 * 3}));
}

TYPED_TEST_P(TensorType, InnerProduct) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorA.assignInitValue(1);
	tensorB.assignInitValue(1);

	ASSERT_EQ(Tensor<TypeParam>::dot(tensorA, tensorB), (8 * 8 * 3) * 1);
}

TYPED_TEST_P(TensorType, Reduce) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 1, 1}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 1, 1}), sizeof(TypeParam));

	ASSERT_NO_THROW(tensorA.reduce());
	ASSERT_NO_THROW(tensorB.reduce());
}

TYPED_TEST_P(TensorType, Reshape) {
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 1, 1}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 1, 1}), sizeof(TypeParam));

		ASSERT_NO_THROW(tensorA.reshape(Shape<uint32_t>({8, 8, 1, 1})));
		ASSERT_NO_THROW(tensorB.reshape(Shape<uint32_t>({8, 8, 1, 1})));
	}
}

TYPED_TEST_P(TensorType, Append) {

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({3, 1}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({3, 1}), sizeof(TypeParam));

		const TypeParam value = (unsigned int)rand() % 100;

		ASSERT_NO_THROW(tensorA.assignInitValue(static_cast<TypeParam>(value)));
		ASSERT_NO_THROW(tensorB.assignInitValue(static_cast<TypeParam>(value)));

		ASSERT_NO_THROW(tensorA.append(tensorB));

		ASSERT_EQ(tensorA.getShape(), Shape<uint32_t>({3, 2}));
		ASSERT_EQ(tensorA.getNrElements(), tensorA.getShape().getNrElements());

		for (size_t i = 0; i < tensorA.getNrElements(); i++) {
			ASSERT_EQ(tensorA.template getValue<TypeParam>(i), static_cast<TypeParam>(value));
		}
	}

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({3, 1}), sizeof(TypeParam));

		// ASSERT_NO_THROW(tensorA.append(1));
	}

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({3, 1}), sizeof(TypeParam));

		// ASSERT_NO_THROW(tensorA.append({1, 1, 1}));
	}
}

TYPED_TEST_P(TensorType, Cast) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({3}), sizeof(TypeParam));
	const Tensor<TypeParam> tensorB(Shape<uint32_t>({3}), sizeof(TypeParam));

	Tensor<TypeParam> ref;
	ASSERT_NO_THROW(tensorA.template cast<int16_t>());

	ASSERT_EQ(tensorA.getElementSize(), sizeof(int16_t));
	ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(int16_t));
	ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(int16_t));

	// ASSERT_NO_THROW(ref = std::move(tensorB.template cast<int16_t>()));
}

TYPED_TEST_P(TensorType, SubSet) {

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

		Tensor<TypeParam> subset = std::move(tensorA.getSubset(0, 1, Shape<uint32_t>({1})));

		// TODO:
		ASSERT_EQ(subset.getShape(), Shape<uint32_t>({8, 8, 3}));

		// ASSERT_THROW(subset.append({0, 1}), std::runtime_error);
	}
}

REGISTER_TYPED_TEST_SUITE_P(TensorType, DefaultConstructor, DefaultType, AssignMove, DataSize, Addition, Subtract,
							MultiplyFactor, ElementCount, FromArray, SetGetValues, Log10, Mean, Flatten, InnerProduct,
							Append, Reduce, Reshape, Cast, SubSet, MatrixMultiplication, Equal, NotEqual);

using TensorPrimitiveDataTypes = ::testing::Types<int16_t, uint16_t, int32_t, uint32_t, long, size_t, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Parameter, TensorType, TensorPrimitiveDataTypes);

// resize
// Axis dim