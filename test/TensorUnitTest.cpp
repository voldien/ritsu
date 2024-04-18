#include "RitsuDef.h"
#include <Tensor.h>
#include <cstdint>
#include <gtest/gtest.h>

using namespace Ritsu;

template <class T> class TensorTest : public ::testing::Test {};
TYPED_TEST_SUITE_P(TensorTest);

TYPED_TEST_P(TensorTest, DefaultConstructor) {
	ASSERT_NO_THROW(Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam)));
	ASSERT_NO_THROW(Tensor<TypeParam> tensor(Shape<uint32_t>({32, 32, 3}), sizeof(TypeParam)));

	ASSERT_THROW(Tensor<TypeParam> tensor(Shape<unsigned int>{}), RuntimeException);
}

TYPED_TEST_P(TensorTest, DefaultType) {
	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
	ASSERT_EQ(tensor.getDType(), typeid(TypeParam));
}

TYPED_TEST_P(TensorTest, PrintNoThrow) {

	{
		Tensor<TypeParam> tensor({32, 32, 3});
		std::stringstream buf;
		ASSERT_NO_THROW(buf << tensor);
	}
	{
		Tensor<TypeParam> tensor({1}); /*	*/
		std::stringstream buf;
		ASSERT_NO_THROW(buf << tensor);
	}
	{
		Tensor<TypeParam> tensor({32, 32, 3}); /*	*/
		std::stringstream buf;
		ASSERT_NO_THROW(buf << tensor);
	}
}

TYPED_TEST_P(TensorTest, AssignMove) {
	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
	void *data = tensor.template getRawData<void>();
	ASSERT_NO_THROW(Tensor<TypeParam> moved = tensor);

	Tensor<TypeParam> moved;
	ASSERT_NO_THROW(moved = std::move(tensor));
	ASSERT_EQ(data, moved.template getRawData<void>());
	ASSERT_EQ(nullptr, tensor.template getRawData<void>());
}

TYPED_TEST_P(TensorTest, Equal) {

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

		tensorB.assignInitValue(1);
		tensorA.assignInitValue(1);

		ASSERT_EQ(tensorA, tensorA);
		ASSERT_EQ(tensorA, tensorB);
	}
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

		tensorB.assignInitValue(1);
		tensorA.assignInitValue(1);

		ASSERT_NO_THROW(Tensor<TypeParam>::equal(tensorA, tensorB));
	}
}

TYPED_TEST_P(TensorTest, NotEqual) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));
	const Tensor<TypeParam> tensorC(Shape<uint32_t>({8, 8, 3}), sizeof(TypeParam));

	tensorB.assignInitValue(1);
	tensorA.assignInitValue(0);

	ASSERT_NE(tensorA, tensorB);
}

TYPED_TEST_P(TensorTest, DataSize) {
	{
		const Tensor<TypeParam> tensor({32, 32, 3});
		ASSERT_EQ(tensor.getDatSize(), tensor.getNrElements() * sizeof(TypeParam));
	}
	{
		const Tensor<TypeParam> tensor({5, 5, 3});
		ASSERT_EQ(tensor.getDatSize(), tensor.getNrElements() * sizeof(TypeParam));
	}
}

TYPED_TEST_P(TensorTest, Addition) {
	/*	*/
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({10}));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({10}));

		tensorA.assignInitValue(static_cast<TypeParam>(0));
		tensorB.assignInitValue(static_cast<TypeParam>(1));

		const Tensor<TypeParam> result = tensorA + tensorB;

		ASSERT_EQ(result, tensorB);
	}
}

TYPED_TEST_P(TensorTest, Subtract) {
	Tensor<TypeParam> tensorA(Shape<uint32_t>({10}), sizeof(TypeParam));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({10}), sizeof(TypeParam));

	tensorA.assignInitValue(1);
	tensorB.assignInitValue(1);

	const Tensor<TypeParam> result = tensorA - tensorB;

	ASSERT_EQ(result, Tensor<TypeParam>::zero(result.getShape()));
}

TYPED_TEST_P(TensorTest, MultiplyFactor) {

	Tensor<TypeParam> expected(Shape<uint32_t>({10}), sizeof(TypeParam));
	const TypeParam value = rand() % 100;
	expected.assignInitValue(value);

	Tensor<TypeParam> tensorA(Shape<uint32_t>({10}), sizeof(TypeParam));
	tensorA.assignInitValue(1);

	const Tensor<TypeParam> result = tensorA * value;
	ASSERT_EQ(result, expected);
}

TYPED_TEST_P(TensorTest, MatrixMultiplication) {

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

		/*	Verify the shape output.	*/
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

		/*	Verify the shape output.	*/
		ASSERT_EQ(result0.getShape(), Shape<uint32_t>({2, 2}));
		ASSERT_EQ(result0.getShape(), result1.getShape());

		for (size_t i = 0; i < tensorA.getNrElements(); i++) {
			ASSERT_EQ(tensorA.template getValue<TypeParam>(i), static_cast<TypeParam>(0));
		}
	}

	/*	*/
	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::zero(Shape<uint32_t>({1006, 6}));
		const Tensor<TypeParam> tensorB = Tensor<TypeParam>::zero(Shape<uint32_t>({6, 1006}));

		const Tensor<TypeParam> result0 = tensorA.transpose() % tensorB.transpose();
		const Tensor<TypeParam> result1 = Tensor<TypeParam>::matrixMultiply(tensorA.transpose(), tensorB.transpose());
		ASSERT_EQ(result0, result1);

		ASSERT_EQ(result0.getShape(), Shape<uint32_t>({6, 6}));
		ASSERT_EQ(result0.getShape(), result1.getShape());
	}
}

TYPED_TEST_P(TensorTest, ElementCount) {

	const Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
	ASSERT_EQ(tensor.getNrElements(), 32 * 32 * 3);

	const Tensor<TypeParam> tensorB({32, 32, 6}, sizeof(TypeParam));
	ASSERT_EQ(tensorB.getNrElements(), 32 * 32 * 6);
}

TYPED_TEST_P(TensorTest, FromArray) {

	{
		/*	*/
		const TypeParam value = 1;
		ASSERT_NO_THROW(Tensor<TypeParam>::fromArray({value, value, value, value, value}));
	}

	/*	No casting*/
	{
		const Tensor<TypeParam> fromArray = std::move(Tensor<TypeParam>::fromArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

		/*	*/
		ASSERT_EQ(fromArray.getNrElements(), 10);
		ASSERT_EQ(fromArray.getShape(), Shape<typename Tensor<TypeParam>::IndexType>({10}));

		/*	*/
		for (size_t i = 0; i < fromArray.getNrElements(); i++) {
			ASSERT_EQ(fromArray.getValue(i), static_cast<TypeParam>(i));
		}
	}

	{
		const TypeParam value = rand() % 100;
		const Tensor<TypeParam> fromArray =
			std::move(Tensor<TypeParam>::fromArray({value, value, value, value, value}));

		/*	*/
		ASSERT_EQ(fromArray.getNrElements(), 5);
		ASSERT_EQ(fromArray.getShape(), Shape<typename Tensor<TypeParam>::IndexType>({5}));

		/*	*/
		for (size_t i = 0; i < fromArray.getNrElements(); i++) {
			ASSERT_EQ(fromArray.getValue(i), value);
		}
	}
}

TYPED_TEST_P(TensorTest, SetGetValues) {
	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

	for (size_t i = 0; i < tensor.getNrElements(); i++) {
		const TypeParam randomValue = static_cast<TypeParam>(rand());
		tensor.template getValue<TypeParam>(i) = static_cast<TypeParam>(randomValue);
		ASSERT_EQ(tensor.template getValue<TypeParam>(i), static_cast<TypeParam>(randomValue));
	}
}

TYPED_TEST_P(TensorTest, Log10) {

	Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

	const TypeParam randomValue = static_cast<TypeParam>(rand());
	tensor.template getValue<TypeParam>(0) = static_cast<TypeParam>(randomValue);

	Tensor<TypeParam>::log10(tensor);

	// ASSERT_EQ(tensor.template getValue<TypeParam>(0), static_cast<TypeParam>(randomValue));
}

TYPED_TEST_P(TensorTest, Mean) {

	{
		Tensor<TypeParam> tensor({32, 10}, sizeof(TypeParam));

		TypeParam value;
		ASSERT_NO_THROW(value = tensor.mean());
	}

	{
		Tensor<TypeParam> tensor({32, 10}, sizeof(TypeParam));

		ASSERT_NO_THROW(Tensor<TypeParam>::mean(tensor, -1));

		const Tensor<TypeParam> result = Tensor<TypeParam>::mean(tensor, -1);
		ASSERT_EQ(result.getShape(), Shape<typename Tensor<TypeParam>::IndexType>({10}));
	}

	{
		Tensor<TypeParam> tensor({32, 10, 10}, sizeof(TypeParam));

		ASSERT_NO_THROW(Tensor<TypeParam>::mean(tensor, -1));

		const Tensor<TypeParam> result = Tensor<TypeParam>::mean(tensor, -1);
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

TYPED_TEST_P(TensorTest, Flatten) {

	{
		Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));
		ASSERT_NO_THROW(tensor.flatten());
	}
	{
		Tensor<TypeParam> tensor({32, 32, 3}, sizeof(TypeParam));

		const TypeParam randomValue = static_cast<TypeParam>(rand());
		tensor.template getValue<TypeParam>(0) = static_cast<TypeParam>(randomValue);

		ASSERT_NO_THROW(tensor.flatten());

		ASSERT_EQ(tensor.template getValue<TypeParam>(0), static_cast<TypeParam>(randomValue));
		ASSERT_EQ(tensor.getShape(), Shape<unsigned int>({32 * 32 * 3}));
	}
}

TYPED_TEST_P(TensorTest, InnerProduct) {

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}));

		tensorA.assignInitValue(1);
		tensorB.assignInitValue(1);

		ASSERT_EQ(tensorA.dot(tensorB, -1), static_cast<TypeParam>((8 * 8 * 3) * 1));
	}

	/*	Improve.*/
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 3}));

		tensorA.assignInitValue(1);
		tensorB.assignInitValue(1);

		ASSERT_EQ(tensorA.dot(tensorB, -1), static_cast<TypeParam>((8 * 8 * 3) * 1));
	}
}

TYPED_TEST_P(TensorTest, Reduce) {

	Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 1, 1}));
	Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 1, 1}));

	ASSERT_NO_THROW(tensorA.reduce());
	ASSERT_NO_THROW(tensorB.reduce());
}

TYPED_TEST_P(TensorTest, Reshape) {
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 1, 1}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({8, 8, 1, 1}), sizeof(TypeParam));

		ASSERT_NO_THROW(tensorA.reshape(Shape<uint32_t>({8, 8, 1, 1})));
		ASSERT_NO_THROW(tensorB.reshape(Shape<uint32_t>({8, 8, 1, 1})));
	}
}

TYPED_TEST_P(TensorTest, Append) {

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({3, 1}), sizeof(TypeParam));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({3, 1}), sizeof(TypeParam));

		const TypeParam value = static_cast<TypeParam>(rand() % 100);

		ASSERT_NO_THROW(tensorA.assignInitValue(static_cast<TypeParam>(value)));
		ASSERT_NO_THROW(tensorB.assignInitValue(static_cast<TypeParam>(value)));

		ASSERT_NO_THROW(tensorA.concatenate(tensorB));

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

TYPED_TEST_P(TensorTest, Cast) {

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({3}), sizeof(TypeParam));

		Tensor<TypeParam> ref;
		ASSERT_NO_THROW(tensorA.template cast<int16_t>());

		ASSERT_EQ(tensorA.getElementSize(), sizeof(int16_t));
		ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(int16_t));
		ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(int16_t));
	}

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({3}), sizeof(TypeParam));

		Tensor<TypeParam> ref;
		ASSERT_NO_THROW(tensorA.template cast<bool>());

		ASSERT_EQ(tensorA.getElementSize(), sizeof(bool));
		ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(bool));
		ASSERT_EQ(tensorA.getDatSize(), tensorA.getNrElements() * sizeof(bool));
	}
}

TYPED_TEST_P(TensorTest, SubSet) {

	/*	Memory.	*/
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}));
		ASSERT_NO_THROW(tensorA.getSubset(0, 1, Shape<uint32_t>({1})));

		Tensor<TypeParam> subset = std::move(tensorA.getSubset(0, 1, Shape<uint32_t>({1})));
		ASSERT_EQ(subset.getShape(), Shape<uint32_t>({1}));
	}

	/*	Shape.	*/
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({8, 8, 3}));
		ASSERT_NO_THROW(tensorA.getSubset({{0}}));

		Tensor<TypeParam> subset0 = std::move(tensorA.getSubset({{0}}));
		// TODO:
		ASSERT_EQ(subset0.getShape(), Shape<uint32_t>({1, 8, 3}));

		ASSERT_NO_THROW(tensorA.getSubset({{0}}).getSubset({{0}}));

		Tensor<TypeParam> subset1 = std::move(tensorA.getSubset({{0}}).getSubset({{0}}));

		// ASSERT_NO_FATAL_FAILURE()
	}
}

TYPED_TEST_P(TensorTest, Transpose) {

	/*	No throw.	*/
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({2, 4}));
		ASSERT_NO_THROW(tensorA.transpose());
	}

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({2, 4}));
		ASSERT_EQ(tensorA.transpose().getShape(), Shape<uint32_t>({4, 2}));
	}

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({2, 4}));
		ASSERT_EQ(tensorA.transpose().transpose().getShape(), Shape<uint32_t>({2, 4}));
	}

	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({2, 4}));
		ASSERT_EQ(tensorA.transpose().transpose().getShape(), Shape<uint32_t>({2, 4}));
	}
}

TYPED_TEST_P(TensorTest, OneShot) {

	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::fromArray({1, 2, 4, 1, 2, 9, 2, 1, 2, 4, 1});
		const TypeParam max = tensorA.max() + 1;

		const Tensor<TypeParam> oneshot = Tensor<TypeParam>::oneShot(tensorA);

		ASSERT_EQ(static_cast<TypeParam>(oneshot.getShape()[-1]), max);
	}
}

TYPED_TEST_P(TensorTest, Max) {

	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::fromArray({1, 2, 4, 1, 2, 9, 2, 1, 2, 4, 1});
		const TypeParam max = tensorA.max();
		ASSERT_EQ(static_cast<TypeParam>(9), max);
	}
}

TYPED_TEST_P(TensorTest, Min) {

	{
		const Tensor<TypeParam> tensorA = Tensor<TypeParam>::fromArray({1, 2, 4, 1, 2, 9, 2, 1, 2, 4, 1});
		const TypeParam min = tensorA.min();
		ASSERT_EQ(static_cast<TypeParam>(1), min);
	}
}

TYPED_TEST_P(TensorTest, AXPY) {

	/*	*/
	{
		Tensor<TypeParam> tensorA(Shape<uint32_t>({12, 12, 1}));
		Tensor<TypeParam> tensorB(Shape<uint32_t>({12, 12, 1}));

		tensorA.assignInitValue(static_cast<TypeParam>(-1));
		tensorB.assignInitValue(static_cast<TypeParam>(1));

		const TypeParam value = static_cast<TypeParam>(rand() % 100);

		ASSERT_NO_THROW(const Tensor<TypeParam> result = (tensorA * value) + tensorB);
	}
}

REGISTER_TYPED_TEST_SUITE_P(TensorTest, DefaultConstructor, DefaultType, PrintNoThrow, AssignMove, DataSize, Addition,
							Subtract, MultiplyFactor, ElementCount, FromArray, SetGetValues, Max, Min, Log10, Mean,
							Flatten, Transpose, InnerProduct, Append, Reduce, Reshape, Cast, SubSet,
							MatrixMultiplication, Equal, NotEqual, OneShot, AXPY);

using TensorPrimitiveDataTypes =
	::testing::Types<bool, int16_t, uint16_t, int32_t, uint32_t, long, size_t, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(Tensor, TensorTest, TensorPrimitiveDataTypes);