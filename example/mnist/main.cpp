#include "Metric.h"
#include "Tensor.h"
#include "layers/GaussianNoise.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Rescaling.h"
#include "mnist_dataset.h"
#include <Ritsu.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	try {
		/*	*/
		const unsigned int batchSize = 1;
		const unsigned int epochs = 128;
		const size_t dataBufferSize = 5;
		const float learningRate = 0.002f;
		bool useBatchNorm = false;

		/*	*/
		Tensor<float> inputResY, inputResTestY;
		Tensor<float> inputDataX, inputTestX;

		/*	*/
		RitsuDataSet::loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
								"t10k-labels.idx1-ubyte", inputDataX, inputResY, inputTestX, inputResTestY);
		/*	*/
		std::cout << "Loaded MNIST Data Set: " << inputDataX.getShape() << " Labels: " << inputResY.getShape()
				  << std::endl;

		/*	Extract data shape.	*/
		Shape<unsigned int> dataShape = inputDataX.getShape().getSubShape(1, 3);
		Shape<unsigned int> resultShape = inputResY.getShape().getSubShape(1, 1);
		const unsigned int output_size = 10;

		inputDataX = inputDataX.cast<float>();
		inputTestX = inputTestX.cast<float>();

		/*	*/
		std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

		Input input0node(dataShape, "input");
		Cast<uint8_t, float> cast2Float;
		Rescaling normalizedLayer(1.0f / 255.0f);

		GuassianNoise noise(0.1, 0.1f);

		Flatten flattenInput("flatten0");
		Flatten flatten("flatten1");

		Dense fw0(256, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer0");
		BatchNormalization BN0;
		Relu relu0;

		Dense fw1 = Dense(128, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer1");
		BatchNormalization BN1;
		Relu relu1;

		Dense fw2 =
			Dense(output_size, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer2");

		Regularization regulation(0.001f, 0.001f);

		Sigmoid outputAct;

		/*	*/
		{
			Layer<float> *lay = &cast2Float(input0node);

			lay = &normalizedLayer(*lay);
			lay = &flattenInput(*lay);

			lay = &noise(*lay);

			lay = &fw0(*lay);
			if (useBatchNorm) {
				lay = &BN0(*lay);
			}
			lay = &relu0(*lay);

			lay = &fw1(*lay);
			if (useBatchNorm) {
				lay = &BN1(*lay);
			}
			lay = &relu1(*lay);

			lay = &fw2(*lay);
			lay = &outputAct(*lay);

			Layer<float> &output = regulation(*lay);

			Model<float> forwardModel({&input0node}, {&output});

			SGD<float> optimizer(learningRate, 0.0);

			MetricAccuracy accuracy;
			MetricMean lossmetric("loss");

			Loss mse_loss(sparse_categorical_crossentropy);
			forwardModel.compile(&optimizer, sparse_categorical_crossentropy,
								 {dynamic_cast<Metric *>(&lossmetric), (Metric *)&accuracy});
			std::cout << forwardModel.summary() << std::endl;

			forwardModel.fit(epochs, inputDataX, inputResY, batchSize);

			Tensor<float> predict = std::move(forwardModel.predict(inputTestX));

			/*	*/
			Tensor<float> predict_result = Tensor<float>::equal(predict, inputResTestY);
			std::cout << predict_result << std::endl;

			// TODO Accuracy.
			std::cout << "Predict " << predict << std::endl;
		}
	} catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
	}

	return EXIT_SUCCESS;
}