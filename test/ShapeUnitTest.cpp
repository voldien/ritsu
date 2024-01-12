#include <Ritsu.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

template <class T> class ShapeType : public ::testing::Test {};
TYPED_TEST_SUITE_P(ShapeType);

TYPED_TEST_P(ShapeType, DefaultConstructor) { ASSERT_NO_THROW(Ritsu::Shape<TypeParam> shape({32, 32, 3})); }

TYPED_TEST_P(ShapeType, SetGetValues) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	shape[0] = 3;
	shape[1] = 24;
	shape[2] = 22;
	ASSERT_EQ(shape[0], 3);
	ASSERT_EQ(shape[1], 24);
	ASSERT_EQ(shape[2], 22);
	ASSERT_EQ(shape.getNrDimensions(), 3);
}

TYPED_TEST_P(ShapeType, Flatten) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	auto& flatten = shape.flatten();
	ASSERT_EQ(flatten, Ritsu::Shape<TypeParam>({32 * 32 * 3}));
	ASSERT_EQ(flatten.getNrDimensions(), 1);
}

TYPED_TEST_P(ShapeType, Reshape) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	shape.reshape({16, 16, 3 * 2 * 2});
	ASSERT_EQ(shape.getNrDimensions(), 3);
}

TYPED_TEST_P(ShapeType, SubShape) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	// auto flatten = shape.getSubShape(1);
	// shape.getNrDimensions();
}

TYPED_TEST_P(ShapeType, Reduce) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	auto &flatten = shape.reduce();
}

TYPED_TEST_P(ShapeType, ComputeIndex) {
	Ritsu::Shape<TypeParam> shape({32, 32, 3});

	// auto &flatten = shape.reduce();
}

REGISTER_TYPED_TEST_SUITE_P(ShapeType, DefaultConstructor, SetGetValues, Flatten, Reshape, SubShape, Reduce,
							ComputeIndex);

using ShapePrimitiveDataTypes = ::testing::Types<uint16_t, uint32_t, size_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Shape, ShapeType, ShapePrimitiveDataTypes);

// Number of elements.
// Axis dim