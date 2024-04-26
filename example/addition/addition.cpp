#include "Metric.h"
#include "Tensor.h"
#include "mnist_dataset.h"
#include <Ritsu.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	/*	*/
	const unsigned int batchSize = 16;
	const unsigned int epochs = 512;
	const size_t dataBufferSize = 5;
	const float learningRate = 0.00005f;
	const size_t trainingDataSize = 2048;
	const float validationSplit = 0.1f;
	bool useBias = false;
	bool useNoise = false;

	const unsigned int output_size = 1;
	const unsigned int dense_size = 2;

	try {
		Input input({2}, "input");
		Dense dense0(dense_size, false);
		Dense outputDense(output_size, false);

		RandomUniformInitializer<float> random(0, 3, 10052);
		Tensor<float> dataX = random(Shape<unsigned int>({trainingDataSize, 2}));

		Tensor<float> testX = random(Shape<unsigned int>({trainingDataSize, 2}));

		/*	Sum.	*/
		Tensor<float> dataY({trainingDataSize, 1});
		for (unsigned int i = 0; i < dataY.getNrElements(); i++) {
			const float value = dataX.getValue({i, 0}) + dataX.getValue({i, 1});
			dataY.getValue(i) = value;
		}

		Layer<float> &output = outputDense(dense0(input));
		SGD<float> optimizer(learningRate, 0.0f);

		MetricAccuracy accuracy;
		Model<float> forwardModel = Model<float>({&input}, {&output});
		MeanSquareError mse_loss = MeanSquareError();

		forwardModel.compile(&optimizer, mse_loss, {dynamic_cast<Metric *>(&accuracy)});
		std::cout << forwardModel.summary() << std::endl;

		Model<float>::History *result = &forwardModel.fit(epochs, dataX, dataY, batchSize, validationSplit, true, true);

		forwardModel.saveWeight("mnist_forward_network_model.weight");

		Tensor<float> predict = forwardModel.predict(testX);

		/*	*/
		// Tensor<float> predict_result = Tensor<float>::equal(predict, inputResTestYF);
		// std::cout << predict_result << std::endl;

		// std::cout << "Predict " << predict << std::endl;

		std::cout << (*result)["loss"] << std::endl;
		std::cout << (*result)["accuracy"] << std::endl;

	} catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}