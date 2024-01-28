#include "layers/Regularization.h"
#include "mnist_dataset.h"
#include <Ritsu.h>
#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	/*	*/
	const unsigned int batchSize = 1;
	const unsigned int epochs = 128;
	const size_t dataBufferSize = 5;
	const float learningRate = 0.002f;

	/*	*/
	Tensor<float> inputResY, inputResTestY;
	Tensor<float> inputDataX, inputTestX;

	/*	*/
	RitsuDataSet::loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
							"t10k-labels.idx1-ubyte", inputDataX, inputResY, inputTestX, inputResTestY);

	std::cout << "Loaded MNIST Data Set: " << inputDataX.getShape() << " Labels: " << inputResY.getShape() << std::endl;

	bool useResnet;

	/*	*/
	{
		Input input0node({32, 32, 1}, "input");

		Conv2D conv2D_0(32, {3, 3}, {2, 2}, "same");
		Relu relu_0;
		BatchNormalization BatchNormalization_0;

		Conv2D conv2D_1(64, {3, 3}, {2, 2}, "same");
		Relu relu_1;
		BatchNormalization BatchNormalization_1;

		Conv2D conv2D_2(128, {3, 3}, {2, 2}, "same");
		Relu relu_2;
		BatchNormalization BatchNormalization_2;

		Flatten flatten0("flatten0");
		Dense output(1);

		Sigmoid sigmoid;
		Regularization regularization(0.1f, 0.2f);

		Layer<float> &outputLayer = regularization(sigmoid(output(
			flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(input0node))))))))));

		Model<float> forwardModel({&input0node}, {&output});

		SGD<float> optimizer(learningRate, 0.0);

		MetricAccuracy accuracy;
		MetricMean lossmetric("loss");

		Loss mse_loss(sparse_categorical_crossentropy);
		forwardModel.compile(&optimizer, mse_loss, {dynamic_cast<Metric *>(&lossmetric), (Metric *)&accuracy});
		std::cout << forwardModel.summary() << std::endl;

		//
		Model<float> model({&input0node}, {&outputLayer});

		Tensor<float> predict = std::move(forwardModel.predict(inputTestX));
		// Compare.
		std::cout << "Predict " << predict << std::endl;
	}
}