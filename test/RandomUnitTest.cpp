#include <Ritsu.h>
#include <gtest/gtest.h>

template <class T> class RandomType : public ::testing::Test {};
TYPED_TEST_SUITE_P(RandomType);

TYPED_TEST_P(RandomType, Uniform) { ASSERT_NO_THROW(Ritsu::RandomUniform<TypeParam> uniform(0, 1)); }

TYPED_TEST_P(RandomType, UniformGet) {
	Ritsu::RandomUniform<TypeParam> uniform(0, 1);
	for (size_t i = 0; i < 1000; i++) {
		TypeParam value;
		ASSERT_NO_THROW(value = uniform.rand());
		ASSERT_LE(value, 1);
		ASSERT_GE(value, 0);
	}
}

TYPED_TEST_P(RandomType, RandomNormal) { ASSERT_NO_THROW(Ritsu::RandomNormal<TypeParam> normal(0, 1)); }

TYPED_TEST_P(RandomType, RandomNormalGet) {
	Ritsu::RandomNormal<TypeParam> uniform(0, 1);
	for (size_t i = 0; i < 1000; i++) {
		ASSERT_NO_THROW(uniform.rand());
	}
}

TYPED_TEST_P(RandomType, RandomBernoulli) { ASSERT_NO_THROW(Ritsu::RandomBernoulli<TypeParam> bernoulli(1)); }

TYPED_TEST_P(RandomType, RandomBernoulliGet) {
	Ritsu::RandomBernoulli<TypeParam> uniform(0, 1);
	for (size_t i = 0; i < 1000; i++) {
		ASSERT_NO_THROW(uniform.rand());
	}
}

REGISTER_TYPED_TEST_SUITE_P(RandomType, Uniform, RandomNormal, RandomBernoulli, UniformGet, RandomNormalGet,
							RandomBernoulliGet);

using RandomDataTypes = ::testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(SW, RandomType, RandomDataTypes);
