#include <Ritsu.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

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

TYPED_TEST_P(ShapeType, PrintNoThrow) {

	{
		Ritsu::Shape<TypeParam> shape; /*	*/
		std::stringstream buf;
		ASSERT_NO_THROW(buf << shape);
	}
	{
		Ritsu::Shape<TypeParam> shape({1}); /*	*/
		std::stringstream buf;
		ASSERT_NO_THROW(buf << shape);
	}
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3}); /*	*/
		std::stringstream buf;
		ASSERT_NO_THROW(buf << shape);
	}
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

	{
		Ritsu::Shape<TypeParam> shape({2, 8, 8, 8, 3});

		const auto &flatten = shape.flatten();
		ASSERT_EQ(flatten, Ritsu::Shape<TypeParam>({2 * 8 * 8 * 8 * 3}));
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

	/*	Check methods are callable.	*/
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		ASSERT_NO_THROW(shape.getSubShape({{0, 10}}));
		ASSERT_NO_THROW(shape.getSubShape({{0, 10}, {1}}));
		ASSERT_NO_THROW(shape.getSubShape({{0, 10}, {1}}));
		ASSERT_NO_THROW(shape.getSubShape({{0}}));
	}

	{
		Ritsu::Shape<TypeParam> shape({6000, 32, 32, 3});

		auto subshape = shape.getSubShape({{0}});
		ASSERT_EQ(subshape, Ritsu::Shape<TypeParam>({1, 32, 32, 3}));

		subshape = shape.getSubShape({{0}, {0}, {0}});
		ASSERT_EQ(subshape, Ritsu::Shape<TypeParam>({1, 1, 1, 3}));
	}

	{
		Ritsu::Shape<TypeParam> shape({6000, 32, 32, 3});

		auto subshape = shape.getSubShape({{0, 1}, {0, 1}});
		ASSERT_EQ(subshape, Ritsu::Shape<TypeParam>({2, 2, 32, 3}));
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

TYPED_TEST_P(ShapeType, Append) {

	/*	*/
	{
		Shape<TypeParam> shape0({3});
		Shape<TypeParam> shape1({3});
		Shape<TypeParam> shape2;

		// TODO:
		ASSERT_NO_THROW(shape2 = shape0 + shape1);

		ASSERT_EQ(shape2, Shape<TypeParam>({6}));
	}

	/*	*/
	{
		Shape<TypeParam> shape0({3, 1});
		Shape<TypeParam> shape1({3, 1});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0 + shape1);

		ASSERT_EQ(shape2, Shape<TypeParam>({3, 2}));
	}

	/*	*/
	{
		Shape<TypeParam> shape0({8, 8, 1});
		Shape<TypeParam> shape1({8, 8, 1});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0 + shape1);

		ASSERT_EQ(shape2, Shape<TypeParam>({8, 8, 2}));
	}

	/*	*/
	{
		Shape<TypeParam> shape0({8, 8, 1});
		Shape<TypeParam> shape1({8, 8, 1});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0.append(shape1));

		ASSERT_EQ(shape2, Shape<TypeParam>({8, 8, 2}));
	}
}

TYPED_TEST_P(ShapeType, Erase) {

	/*	*/
	{
		Shape<TypeParam> shape0({6});
		Shape<TypeParam> shape1({3});
		Shape<TypeParam> shape2;

		// TODO:
		ASSERT_NO_THROW(shape2 = shape0 - shape1);

		ASSERT_EQ(shape2, Shape<TypeParam>({3}));
	}

	/*	*/
	{
		Shape<TypeParam> shape0({3, 2});
		Shape<TypeParam> shape1({3, 1});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0 - shape1);

		ASSERT_EQ(shape2, Shape<TypeParam>({3, 1}));
	}

	/*	*/
	{
		Shape<TypeParam> shape0({8, 8, 3});
		Shape<TypeParam> shape1({8, 8, 2});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0 - shape1);

		ASSERT_EQ(shape2, Shape<TypeParam>({8, 8, 1}));
	}
	/*	*/
	{
		Shape<TypeParam> shape0({8, 8, 3});
		Shape<TypeParam> shape1({8, 8, 2});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0.erase(shape1));

		ASSERT_EQ(shape2, Shape<TypeParam>({8, 8, 1}));
	}
}

TYPED_TEST_P(ShapeType, Insert) {

	/*	*/
	{
		Shape<TypeParam> shape0({8, 8, 3});
		Shape<TypeParam> shape1({3});
		Shape<TypeParam> shape2;
		// TODO:
		ASSERT_NO_THROW(shape2 = shape0.insert(0, shape1));

		ASSERT_EQ(shape2, Shape<TypeParam>({3, 8, 8, 3}));
	}
}

TYPED_TEST_P(ShapeType, Equality) {

	/*	*/
	{
		Shape<TypeParam> shape0({3});
		Shape<TypeParam> shape1({3});

		ASSERT_EQ(shape0, shape1);
		ASSERT_EQ(shape0, shape0);
		ASSERT_EQ(shape1, shape1);
	}

	/*	*/
	{
		Shape<TypeParam> shape0({28, 28, 3});
		Shape<TypeParam> shape1({28, 28, 3});

		ASSERT_EQ(shape0, shape1);
		ASSERT_EQ(shape0, shape0);
		ASSERT_EQ(shape1, shape1);
	}

	/*	*/
	{
		Shape<TypeParam> shape0({3});
		Shape<TypeParam> shape1({6});

		ASSERT_NE(shape0, shape1);
		ASSERT_NE(shape1, shape0);
	}

	/*	*/
	{
		Shape<TypeParam> shape0({3, 1});
		Shape<TypeParam> shape1({3});

		ASSERT_EQ(shape0, shape1);
		ASSERT_EQ(shape1, shape0);
	}
}

TYPED_TEST_P(ShapeType, ComputeIndex) {
	{
		Ritsu::Shape<TypeParam> shape({32, 32, 3});

		ASSERT_EQ(Ritsu::Shape<TypeParam>::computeIndex({0}, shape), 0);
		ASSERT_EQ(Ritsu::Shape<TypeParam>::computeIndex({1}, shape), 1);

		ASSERT_EQ(Ritsu::Shape<TypeParam>::computeIndex({16, 16, 0}, shape), 32 * 16 + 16);
	}
}

TYPED_TEST_P(ShapeType, MemoryIndexOrder) {
	/*	*/
	{
		Shape<TypeParam> shape({3});
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 0, 0), 0);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 1, 0), 1);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 2, 0), 2);
	}
	{
		Shape<TypeParam> shape({32, 32, 3});
		/*	*/
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 0, 1), 0);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 0, 2), 0);

		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 1, 1), 32);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 2, 1), 64);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 32, 1), 1);

		/*	*/
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 1, 2), 1024);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 2, 2), 2048);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 3, 2), 1);
		ASSERT_EQ(Shape<TypeParam>::getIndexMemoryOffset(shape, 32 * 3, 2), 32);
	}
}

REGISTER_TYPED_TEST_SUITE_P(ShapeType, DefaultConstructor, DimIndexOrder, SetGetValues, PrintNoThrow, Flatten, Reshape,
							SubShape, Reduce, ComputeIndex, Append, Erase, Insert, Equality, MemoryIndexOrder);

using ShapePrimitiveDataTypes = ::testing::Types<int16_t, uint16_t, int32_t, uint32_t, size_t, ssize_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Shape, ShapeType, ShapePrimitiveDataTypes);