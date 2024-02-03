#include "core/Shape.h"
#include <Ritsu.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <tuple>

using namespace Ritsu;

template <class T> class ShapeType : public ::testing::Test {};
TYPED_TEST_SUITE_P(ShapeType);

TYPED_TEST_P(ShapeType, DefaultConstructor) { ASSERT_NO_THROW(Ritsu::Shape<TypeParam> shape({32, 32, 3})); }

TYPED_TEST_P(ShapeType, DimIndexOrder) {
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});
		ASSERT_EQ(shape[0], 32);
		ASSERT_EQ(shape[1], 32);
		ASSERT_EQ(shape[2], 3);

		ASSERT_EQ(shape.getNrDimensions(), 3);
	}

	{
		Ritsu::Shape<TypeParam> shape({64, 32, 32, 3});
		ASSERT_EQ(shape[0], 64);
		ASSERT_EQ(shape[1], 32);
		ASSERT_EQ(shape[2], 32);
		ASSERT_EQ(shape[3], 3);

		ASSERT_EQ(shape.getNrDimensions(), 4);

		/*	Modular index.	*/
		ASSERT_EQ(shape[-1], 3);
		ASSERT_EQ(shape[-2], 32);
		ASSERT_EQ(shape[-3], 32);
		ASSERT_EQ(shape[-4], 64);
	}
}

TYPED_TEST_P(ShapeType, SetGetValues) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	shape[0] = 3;
	shape[1] = 24;
	shape[2] = 22;
	ASSERT_EQ(shape[0], 3);
	ASSERT_EQ(shape[1], 24);
	ASSERT_EQ(shape[2], 22);
	ASSERT_EQ(shape.getNrDimensions(), 3);
	/*	*/
	ASSERT_EQ(shape[3], shape[0]);
}

TYPED_TEST_P(ShapeType, Flatten) {
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		const auto &flatten = shape.flatten();
		ASSERT_EQ(flatten, Ritsu::Shape<TypeParam>({32 * 32 * 3}));
		ASSERT_EQ(flatten.getNrDimensions(), 1);
	}
	{
		Ritsu::Shape<TypeParam> shape({8 * 32, 32, 3});

		const auto &flatten = shape.flatten();
		ASSERT_EQ(flatten, Ritsu::Shape<TypeParam>({8 * 32 * 32 * 3}));
		ASSERT_EQ(flatten.getNrDimensions(), 1);
	}
}

TYPED_TEST_P(ShapeType, Reshape) {
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		ASSERT_NO_THROW(shape.reshape({16, 16, 3 * 2 * 2}));
		ASSERT_EQ(shape.getNrDimensions(), 3);
	}

	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		ASSERT_NO_THROW(shape.reshape({1, 1, 32, 32, 3}));
		ASSERT_EQ(shape.getNrDimensions(), 5);
	}
}

TYPED_TEST_P(ShapeType, SubShape) {
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		auto subshape = shape.getSubShape(0);

		ASSERT_EQ(subshape, Ritsu::Shape<TypeParam>({32, 3}));
	}

	{
		Ritsu::Shape<TypeParam> shape({32, 3});

		auto subshape = shape.getSubShape(0);

		ASSERT_EQ(subshape, Ritsu::Shape<TypeParam>({32, 3}));
	}
}

TYPED_TEST_P(ShapeType, Reduce) {
	{
		Ritsu::Shape<TypeParam> shape({1, 32, 32, 3});
		const auto &reduce = shape.reduce();
		ASSERT_EQ(reduce, Shape<TypeParam>({32, 32, 3}));
	}

	{
		Ritsu::Shape<TypeParam> shape({1, 1, 32, 32, 3});
		const auto &reduce = shape.reduce();
		ASSERT_EQ(reduce, Shape<TypeParam>({32, 32, 3}));
	}

	{
		Ritsu::Shape<TypeParam> shape({1, 1, 32, 32, 1});
		const auto &reduce = shape.reduce();
		ASSERT_EQ(reduce, Shape<TypeParam>({32, 32})); // TODO: determine if correct.
	}
}

// TODO:
TYPED_TEST_P(ShapeType, ComputeIndex) {
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		ASSERT_EQ(Ritsu::Shape<TypeParam>::computeIndex({0}), 0);
		ASSERT_EQ(Ritsu::Shape<TypeParam>::computeIndex({10}), 10);

		ASSERT_EQ(Ritsu::Shape<TypeParam>::computeIndex({32, 32, 1}), 10);
	}
}

TYPED_TEST_P(ShapeType, Append) {

	/*	*/
	{
		Shape<uint32_t> a({3});
		Shape<uint32_t> b({3});
		Shape<uint32_t> c;

		// TODO:
		ASSERT_NO_THROW(c = a + b);

		ASSERT_EQ(c, Shape<uint32_t>({6}));
	}

	/*	*/
	{
		Shape<uint32_t> a({3, 1});
		Shape<uint32_t> b({3, 1});
		Shape<uint32_t> c;
		// TODO:
		ASSERT_NO_THROW(c = a + b);

		ASSERT_EQ(c, Shape<uint32_t>({3, 2}));
	}

	/*	*/
	{
		Shape<uint32_t> a({8, 8, 1});
		Shape<uint32_t> b({8, 8, 1});
		Shape<uint32_t> c;
		// TODO:
		ASSERT_NO_THROW(c = a + b);

		ASSERT_EQ(c, Shape<uint32_t>({8, 8, 2}));
	}
}

TYPED_TEST_P(ShapeType, Erase) {

	/*	*/
	{
		Shape<uint32_t> a({6});
		Shape<uint32_t> b({3});
		Shape<uint32_t> c;

		// TODO:
		ASSERT_NO_THROW(c = a - b);

		ASSERT_EQ(c, Shape<uint32_t>({3}));
	}

	/*	*/
	{
		Shape<uint32_t> a({3, 2});
		Shape<uint32_t> b({3, 1});
		Shape<uint32_t> c;
		// TODO:
		ASSERT_NO_THROW(c = a - b);

		ASSERT_EQ(c, Shape<uint32_t>({3, 1}));
	}

	/*	*/
	{
		Shape<uint32_t> a({8, 8, 3});
		Shape<uint32_t> b({8, 8, 2});
		Shape<uint32_t> c;
		// TODO:
		ASSERT_NO_THROW(c = a - b);

		ASSERT_EQ(c, Shape<uint32_t>({8, 8, 1}));
	}
}

TYPED_TEST_P(ShapeType, Equality) {

	/*	*/
	{
		Shape<uint32_t> a({3});
		Shape<uint32_t> b({3});

		ASSERT_EQ(a, b);
		ASSERT_EQ(a, a);
		ASSERT_EQ(b, b);
	}

	/*	*/
	{
		Shape<uint32_t> a({28, 28, 3});
		Shape<uint32_t> b({28, 28, 3});

		ASSERT_EQ(a, b);
		ASSERT_EQ(a, a);
		ASSERT_EQ(b, b);
	}

	/*	*/
	{
		Shape<uint32_t> a({3});
		Shape<uint32_t> b({6});

		ASSERT_NE(a, b);
		ASSERT_NE(b, a);
	}
}

REGISTER_TYPED_TEST_SUITE_P(ShapeType, DefaultConstructor, DimIndexOrder, SetGetValues, Flatten, Reshape, SubShape,
							Reduce, ComputeIndex, Append, Erase, Equality);

using ShapePrimitiveDataTypes = ::testing::Types<int16_t, uint16_t, int32_t, uint32_t, size_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Shape, ShapeType, ShapePrimitiveDataTypes);

// Axis dim