#include <Ritsu.h>
#include <gtest/gtest.h>
#include <tuple>

using namespace Ritsu;

class ModelTest : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, Ritsu::Shape<uint32_t>>> {
  public:
	// TODO: setup
};

TEST_P(ModelTest, Setup) {
	auto [x, y, expected] = GetParam();

	Input input0node(expected, "input");
	Cast<uint8_t, float> cast2Float;
	Rescaling normalizedLayer(1.0f / 255.0f);
	GuassianNoise noise(0.1, 0.1f);

	Layer<float> &output = noise(normalizedLayer(cast2Float(input0node)));

	Model<float> *forwardModel = nullptr;
	ASSERT_NO_THROW(forwardModel = new Model<float>({&input0node}, {&output}));
	EXPECT_EQ(output.getShape(), expected);
	ASSERT_NO_THROW(delete forwardModel);
}

TEST_P(ModelTest, Compile) {
	auto [x, y, expected] = GetParam();

	Input input0node(expected, "input");
	Cast<uint8_t, float> cast2Float;
	Rescaling normalizedLayer(1.0f / 255.0f);
	GuassianNoise noise(0.1, 0.1f);

	Layer<float> &output = noise(normalizedLayer(cast2Float(input0node)));

	Model<float> *forwardModel = nullptr;
	ASSERT_NO_THROW(forwardModel = new Model<float>({&input0node}, {&output}));
	EXPECT_EQ(output.getShape(), expected);
	ASSERT_NO_THROW(delete forwardModel);
}

TEST_P(ModelTest, Fit) {
	auto [x, y, expected] = GetParam();

	Input input0node(expected, "input");
	Cast<uint8_t, float> cast2Float;
	Rescaling normalizedLayer(1.0f / 255.0f);
	GuassianNoise noise(0.1, 0.1f);

	Layer<float> &output = noise(normalizedLayer(cast2Float(input0node)));

	Model<float> *forwardModel = nullptr;
	ASSERT_NO_THROW(forwardModel = new Model<float>({&input0node}, {&output}));
	EXPECT_EQ(output.getShape(), expected);
	ASSERT_NO_THROW(delete forwardModel);
}

TEST_P(ModelTest, Predict) {
	auto [x, y, expected] = GetParam();

	Input input0node(expected, "input");
	Cast<uint8_t, float> cast2Float;
	Rescaling normalizedLayer(1.0f / 255.0f);
	GuassianNoise noise(0.1, 0.1f);

	Layer<float> &output = noise(normalizedLayer(cast2Float(input0node)));

	Model<float> *forwardModel = nullptr;
	ASSERT_NO_THROW(forwardModel = new Model<float>({&input0node}, {&output}));
	EXPECT_EQ(output.getShape(), expected);
	ASSERT_NO_THROW(delete forwardModel);
}

INSTANTIATE_TEST_SUITE_P(Model, ModelTest,
						 ::testing::Values(std::make_tuple(16, 32, Ritsu::Shape<uint32_t>({16 * 32}))));