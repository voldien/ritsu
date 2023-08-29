#include "Metric.h"
#include <Ritsu.h>

#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

void loadMNIST(const std::string &imagePath, Tensor &dataX, Tensor &dataY) {
	/*	*/
	/*	*/
}

int main(int argc, const char **argv) {

	/*	*/
	const unsigned int batchSize = 1;
	const unsigned int output_size = 10;
	const unsigned int epochs = 128;
	Shape<unsigned int> dataShape({32, 32, 1});
	Shape<unsigned int> resultShape({10});

	const size_t dataBufferSize = 100;
	Tensor inputResY({dataBufferSize, output_size}, 4);
	Tensor inputDataX({dataBufferSize, 32, 32, 1}, 4);

	loadMNIST("", inputDataX, inputResY);

	Input input0node(dataShape, "input");

	Flatten flattenInput("flatten0");
	Flatten flatten("flatten1");

	Relu relu0;
	Relu relu1;

	Dense fw0(256, true, "layer0");
	BatchNormalization BN0;
	Dense fw1 = Dense(128, true, "layer1");
	BatchNormalization BN1;
	Dense fw2 = Dense(output_size, true, "layer2");

	Sigmoid outputAct;

	GuassianNoise noiseLayer(0.05, 0.05f, "noise");

	Layer<float> &output = outputAct(fw2(relu1(BN1(fw1(relu0(BN0(fw0(flattenInput(input0node)))))))));

	Model<float> forwardModel({&input0node}, {&output});

	SGD<float> optimizer(0.002f, 0.0);

	MetricAccuracy accuracy;
	MetricMean lossmetric;

	Loss mse_loss(loss_mse);
	forwardModel.compile(&optimizer, loss_cross_entropy, {(Metric *)&mse_loss, (Metric *)&accuracy});
	std::cout << forwardModel.summary() << std::endl;

	forwardModel.fit(epochs, inputDataX, inputResY, batchSize);
	Tensor predict = std::move(forwardModel.predict(inputDataX));
	std::cout << "Predict " << predict << std::endl;

	return EXIT_SUCCESS;
}