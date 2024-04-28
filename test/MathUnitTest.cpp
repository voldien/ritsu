#include <Ritsu.h>
#include <gtest/gtest.h>

using namespace Ritsu;

class ClampTest : public ::testing::TestWithParam<std::tuple<double, double, double, double>> {};

TEST_P(ClampTest, Values) {
	auto [x, min, max, expected] = GetParam();
	auto clampedValue = Math::clamp(x, min, max);

	EXPECT_FLOAT_EQ(clampedValue, expected);
}

INSTANTIATE_TEST_SUITE_P(Math, ClampTest,
						 ::testing::Values(std::make_tuple(5, 3, 4, 4), std::make_tuple(1, 3, 4, 3),
										   std::make_tuple(-1000, 3, 4, 3)));

class MaxTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {};

TEST_P(MaxTest, Values) {
	auto [x, y, expected] = GetParam();
	auto maxValue = Math::max(x, y);

	EXPECT_FLOAT_EQ(maxValue, expected);
}

INSTANTIATE_TEST_SUITE_P(Math, MaxTest,
						 ::testing::Values(std::make_tuple(5, 3, 5), std::make_tuple(5, 3, 5),
										   std::make_tuple(5, 3, 5)));

class MinTest : public ::testing::TestWithParam<std::tuple<double, double, double>> {};

TEST_P(MinTest, Values) {
	auto [x, y, expected] = GetParam();
	auto minValue = Math::min(x, y);

	EXPECT_FLOAT_EQ(minValue, expected);
}

INSTANTIATE_TEST_SUITE_P(Math, MinTest,
						 ::testing::Values(std::make_tuple(5, 3, 3), std::make_tuple(5, 3, 3),
										   std::make_tuple(5, 3, 3)));

TEST(Math, PowerOf2_Found) {

	for (unsigned int i = 0; i < 31; i++) {
		ASSERT_TRUE(Math::IsPowerOfTwo<unsigned long int>(1 << i));
	}
}

TEST(Math, PowerOf2_Next_Found) {

	for (unsigned int i = 0; i < 30; i++) {
		const auto po2 = Math::NextPowerOfTwo<unsigned long int>((1 << i) + 1);

		ASSERT_TRUE(Math::IsPowerOfTwo(po2));

		ASSERT_EQ(po2, static_cast<unsigned long int>(1 << (i + 1)));
	}
}

class SumTest : public ::testing::TestWithParam<std::tuple<std::vector<float>, float>> {};

TEST_P(SumTest, Values) {
	auto [x, expected] = GetParam();
	auto sum = Math::sum(x);

	EXPECT_NEAR(sum, expected, 10e-6);
}

INSTANTIATE_TEST_SUITE_P(Math, SumTest,
						 ::testing::Values(std::make_tuple(std::vector<float>{1, 2, 3, 4, 5}, 15),
										   std::make_tuple(std::vector<float>{5, 5, 5, 5, 5}, 25),
										   std::make_tuple(std::vector<float>{-5, 5, -5, 5, 5}, 5)));

class ProductTest : public ::testing::TestWithParam<std::tuple<std::vector<float>, float>> {};

TEST_P(ProductTest, Values) {
	auto [x, expected] = GetParam();
	auto sum = Math::product(x);

	EXPECT_NEAR(sum, expected, 10e-6);
}

INSTANTIATE_TEST_SUITE_P(Math, ProductTest,
						 ::testing::Values(std::make_tuple(std::vector<float>{1, 2, 3, 4, 5}, 120),
										   std::make_tuple(std::vector<float>{5, 5, 5, 5, 5}, 3125),
										   std::make_tuple(std::vector<float>{-1, 2, -10, 5, 5}, 500)));

class GuassianDistributionTest : public ::testing::TestWithParam<std::tuple<float, float, std::vector<float>>> {};

TEST_P(GuassianDistributionTest, Values) {
	auto [theta, standard_deviation, expected] = GetParam();
	std::vector<float> guass(expected.size());

	Math::guassian(guass, expected.size(), theta, standard_deviation);

	ASSERT_EQ(expected.size(), guass.size());

	const float sum = Math::sum<float>(guass);
	EXPECT_NEAR(sum, 1.0f, 0.015f);
}

INSTANTIATE_TEST_SUITE_P(Math, GuassianDistributionTest,
						 ::testing::Values(std::make_tuple(0.0, 0.5, std::vector<float>{1, 2, 2, 5, 3, 4, 5, 5, 5}),
										   std::make_tuple(0.0, 0.5, std::vector<float>{5, 5, 2, 5, 5, 5, 5, 5, 5})));

TEST(Math, Distrubtion) {
	/*	Guassian distribution.	*/
	const float mean = 0.0f;
	const int num_guass = 30;
	const int num_total_guass = num_guass * 2 + 1;
	/*	*/
	std::vector<float> guassian(num_total_guass);
	Math::guassian(guassian, num_guass, mean, 1.0f);
	/*	*/
	ASSERT_EQ(guassian.size(), num_total_guass);
	float sum = Math::sum(guassian);
	ASSERT_NEAR(sum, 1.0f, 0.0005f);

	Math::guassian(guassian, num_guass, mean, 0.1f);
}

class AlignmentTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(AlignmentTest, Values) {
	auto [x, align, expected] = GetParam();
	auto clampedValue = Math::align<size_t>(x, align);

	EXPECT_FLOAT_EQ(clampedValue, expected);
}

INSTANTIATE_TEST_SUITE_P(Math, AlignmentTest,
						 ::testing::Values(std::make_tuple(23, 64, 64), std::make_tuple(244, 128, 256),
										   std::make_tuple(300, 64, 320)));

class MeanTest : public ::testing::TestWithParam<std::tuple<const std::vector<float>, float>> {};

TEST_P(MeanTest, Values) {
	auto [x, expected] = GetParam();

	const float mean = Math::mean(x);

	EXPECT_FLOAT_EQ(mean, expected);
}
INSTANTIATE_TEST_SUITE_P(
	Math, MeanTest,
	::testing::Values(
		std::make_tuple(std::vector<float>({9, 10, 12, 13, 13, 13, 15, 15, 16, 16, 18, 22, 23, 24, 24, 25}), 16.75f),
		std::make_tuple(std::vector<float>({-10, 1, 4, 5, 10, 20, 70}), 14.285714285714f),
		std::make_tuple(std::vector<float>({-10}), -10)));

class VarianceTest : public ::testing::TestWithParam<std::tuple<std::vector<float>, float>> {};

TEST_P(VarianceTest, Values) {
	auto [x, expected] = GetParam();

	const float mean = Math::mean<float>(x);
	const float variance = Math::variance<float>(x, mean);

	EXPECT_FLOAT_EQ(variance, expected);
}
INSTANTIATE_TEST_SUITE_P(Math, VarianceTest,
						 ::testing::Values(std::make_tuple(std::vector<float>({1, 2, 3, 4, 5, 5, 5}), 2.6190476f),
										   std::make_tuple(std::vector<float>({-10, 20, -15, 20, 2, -100}),
														   1996.1667f)));

class StandardDeviationTest : public ::testing::TestWithParam<std::tuple<std::vector<float>, float>> {};

TEST_P(StandardDeviationTest, Values) {
	auto [x, expected] = GetParam();

	const float mean = Math::mean<float>(x);
	const float variance = Math::standardDeviation<float>(x, mean);

	EXPECT_FLOAT_EQ(variance, expected);
}
INSTANTIATE_TEST_SUITE_P(Math, StandardDeviationTest,
						 ::testing::Values(std::make_tuple(std::vector<float>({1, 2, 3, 4, 5, 5, 5}), 1.6183472f),
										   std::make_tuple(std::vector<float>({-10, 20, -15, 20, 2, -100}),
														   44.678481f)));

// covariance