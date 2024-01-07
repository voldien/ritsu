#include <Ritsu.h>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

template <class T> class ShapeType : public ::testing::Test {};
TYPED_TEST_SUITE_P(ShapeType);

TYPED_TEST_P(ShapeType, DefaultConstructor) { 
    //ASSERT_NO_THROW(Ritsu::Shape<TypeParam> shape({32, 32, 3})); 
    //ASSERT_NO_THROW(Ritsu::Shape<TypeParam> shape({32, 32, 3})); 
}

TYPED_TEST_P(ShapeType, SetGetValues) {
	//Ritsu::Shape<TypeParam> shape({32, 32, 3});
	//shape.getNrDimensions();
}

REGISTER_TYPED_TEST_SUITE_P(ShapeType, DefaultConstructor, SetGetValues);

using ShapePrimitiveDataTypes = ::testing::Types<uint16_t, uint32_t, size_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Shape, ShapeType, ShapePrimitiveDataTypes);

// Flatten size
// Resize size
// Number of elements.
// Sub shape
// Axis dim
// compute index

/*  */
// template <class T> class LayerType : public ::testing::Test {};
// TYPED_TEST_SUITE_P(LayerType);
//
// TYPED_TEST_P(LayerType, DefaultConstructor) { Ritsu::Layer<TypeParam> layer; }
//
// TYPED_TEST_P(LayerType, SetGetValues) {}
//
// REGISTER_TYPED_TEST_SUITE_P(LayerType, DefaultConstructor, SetGetValues);
//
// using LayerPrimitiveDataTypes = ::testing::Types<uint16_t, uint32_t, size_t>;
// INSTANTIATE_TYPED_TEST_SUITE_P(Layer, LayerType, LayerPrimitiveDataTypes);