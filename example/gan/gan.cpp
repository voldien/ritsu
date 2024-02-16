#include "Metric.h"
#include "Tensor.h"
#include "layers/Conv2D.h"
#include "layers/LeakyRelu.h"
#include "layers/Regularization.h"
#include "layers/Reshape.h"
#include "layers/Tanh.h"
#include "layers/UpSampling2D.h"
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
	const unsigned int batchSize = 1;
	const unsigned int epochs = 128;
	const size_t dataBufferSize = 5;
	const float learningRate = 0.002f;

	Shape<unsigned int> generatorInputSize; // = {64, 1};

	/*	*/
	Tensor<uint8_t> inputResY;
	Tensor<uint8_t> inputResTestY;

	Tensor<float> inputDataX;
	Tensor<float> inputTestX;

	/*	*/
	RitsuDataSet::loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
							"t10k-labels.idx1-ubyte", inputDataX, inputResY, inputTestX, inputResTestY);
	/*	*/
	std::cout << "Loaded MNIST Data Set: " << inputDataX.getShape() << " Labels: " << inputResY.getShape() << std::endl;

	/*	*/
	Shape<unsigned int> dataShape = inputDataX.getShape().getSubShapeMem(1, 3);
	Shape<unsigned int> resultShape = inputResY.getShape().getSubShapeMem(1, 1);
	const unsigned int output_size = 1;

	/*	*/
	std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

	Input input0node(dataShape, "input");

	Flatten flattenInput("flatten0");
	Flatten flatten("flatten1");

	Dense fw0(256, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer0");
	BatchNormalization BN0;
	Relu relu0;

	Dense fw1 = Dense(128, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer1");
	BatchNormalization BN1;
	Relu relu1;

	Dense fw2 = Dense(output_size, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer2");

	Regularization regulation(0.001f, 0.001f);

	Sigmoid outputAct;

	Input inputG(generatorInputSize, "");

	Dense base0(49);
	Shape<unsigned int> reshapeShape;
	Reshape reshape(reshapeShape);

	/*	Process to 14.	*/
	Conv2D conv_0(32, {1, 1}, {1, 1}, "same");
	LeakyRelu leaky_0(0.2f);
	UpSampling2D<float> upscale_0(2);

	/*	Process to 28.	*/
	Conv2D conv_1(32, {1, 1}, {1, 1}, "same");
	LeakyRelu leaky_1(0.2f);
	UpSampling2D<float> upscale_1(2);

	Tanh outputAc;

	/*	*/
	{
		Layer<float> &DiscOutput =
			regulation(outputAct(fw2(relu1(BN1(fw1(relu0(BN0(fw0(flattenInput(input0node))))))))));
		Model<float> discriminatorModel({&input0node}, {&DiscOutput});

		Layer<float> &GenOutput = conv_0(reshape(base0(inputG)));

		SGD<float> optimizer(learningRate, 0.0);

		MetricAccuracy accuracy;
		MetricMean lossmetric("loss");

		Loss mse_loss(loss_mse);
		discriminatorModel.compile(&optimizer, mse_loss, {dynamic_cast<Metric *>(&lossmetric), (Metric *)&accuracy});
		std::cout << discriminatorModel.summary() << std::endl;

		//	discriminatorModel.fit(epochs, inputDataX, inputResY, batchSize);

		Tensor<float> predict = std::move(discriminatorModel.predict(inputTestX));
		// TODO Compare.
		// TODO Accuracy.
		std::cout << "Predict " << predict << std::endl;
	}

	return EXIT_SUCCESS;
}