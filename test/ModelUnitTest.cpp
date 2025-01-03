#include "Loss.h"
#include "core/Initializers.h"
#include "core/Shape.h"
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

	/*	*/
	Input input0node(expected, "input");
	Cast<uint8_t, float> cast2Float;
	Rescaling normalizedLayer(1.0f / 255.0f);
	GuassianNoise noise(0.1, 0.1f);

	/*	*/
	Layer<float> &output = noise(normalizedLayer(cast2Float(input0node)));

	Model<float> *forwardModel = nullptr;
	ASSERT_NO_THROW(forwardModel = new Model<float>({&input0node}, {&output}));
	ASSERT_EQ(output.getShape(), expected);
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
	ASSERT_EQ(output.getShape(), expected);

	SGD<float> optimizer(0.1, 0.0);

	MetricAccuracy accuracy;

	// Loss& mse_loss(sparse_categorical_crossentropy);
	// forwardModel->compile(&optimizer, sparse_categorical_crossentropy, {dynamic_cast<Metric *>(&accuracy)});

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
	ASSERT_EQ(output.getShape(), expected);
	// forwardModel->fit(1, const Tensor<float> &inputData, const Tensor<float> &expectedData)
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

TEST(ModelTest, DenseAddition) {

	Input input({2}, "input");
	Dense dense0(2, false);
	Dense outputDense(1, false);

	RandomUniformInitializer<float> random(0, 2, 10052);
	Tensor<float> dataX = random(Shape<unsigned int>({128, 2}));

	/*	Sum.	*/
	Tensor<float> dataY({128, 1});
	for (unsigned int i = 0; i < dataY.getNrElements(); i++) {

		const float value = dataX.getValue({i, 0}) + dataX.getValue({i, 1});
		dataY.getValue(i) = value;
	}

	Layer<float> &output = outputDense(dense0(input));
	SGD<float> optimizer(0.0001f, 0.0);

	MetricAccuracy accuracy;
	Model<float> forwardModel = Model<float>({&input}, {&output});
	MeanSquareError mse_loss = MeanSquareError();
	forwardModel.compile(&optimizer, mse_loss, {dynamic_cast<Metric *>(&accuracy)});

	Model<float>::History *result = nullptr;
	ASSERT_NO_FATAL_FAILURE(result = &forwardModel.fit(8, dataX, dataY, 1, 0, false, false));

	// EXPECT_NEAR((*result)["loss"].getValue((*result)["loss"].getNrElements() - 1), 0, 0.2);
	// EXPECT_NEAR((*result)["accuracy"].getValue((*result)["accuracy"].getNrElements() - 1), 1, 0.01);
}
