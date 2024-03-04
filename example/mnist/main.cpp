#include "Metric.h"
#include "Tensor.h"
#include "layers/GaussianNoise.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Rescaling.h"
#include "layers/Sigmoid.h"
#include "layers/SoftMax.h"
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
		const float learningRate = 0.01f;
		bool useBatchNorm = false;

		/*	*/
		Tensor<uint8_t> inputResY;
		Tensor<uint8_t> inputResTestY;

		Tensor<uint8_t> inputDataX;
		Tensor<uint8_t> inputTestX;

		/*	*/
		RitsuDataSet::loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
								"t10k-labels.idx1-ubyte", inputDataX, inputResY, inputTestX, inputResTestY);
		/*	*/
		std::cout << "Loaded MNIST Data Set: " << inputDataX.getShape() << " Labels: " << inputResY.getShape()
				  << std::endl;

		/*	*/
		inputResY = Tensor<uint8_t>::oneShot(inputResY);
		inputResTestY = Tensor<uint8_t>::oneShot(inputResTestY);

		/*	*/
		const Tensor<float> inputResYF = inputResY.cast<float>();
		const Tensor<float> inputResTestYF = inputResTestY.cast<float>();

		/*	Extract data shape.	*/
		Shape<unsigned int> dataShape = inputDataX.getShape().getSubShapeMem(1, 3);
		Shape<unsigned int> resultShape = inputResY.getShape().getSubShapeMem(1, 1);
		const unsigned int output_size = 10;

		const Tensor<float> inputDataXF = inputDataX.cast<float>();
		const Tensor<float> inputTestXF = inputTestX.cast<float>();

		/*	*/
		std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

		Input input0node(dataShape, "input");
		Cast<uint8_t, float> cast2Float;
		Rescaling normalizedLayer(1.0f / 255.0f);

		GuassianNoise noise(0.1, 0.1f);

		Flatten flattenInput("flatten0");
		Flatten flatten("flatten1");

		Dense fw0(128, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer0");
		BatchNormalization BN0;
		Relu relu0;

		Dense fw1 = Dense(64, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer1");
		BatchNormalization BN1;
		Relu relu1;

		Dense fw2 =
			Dense(output_size, true, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer2");

		Regularization regulation(0.00005f, 0.000f);

		Sigmoid sigmoid;
		SoftMax outputAct;

		/*	*/
		{
			Layer<float> *lay = &normalizedLayer(input0node);
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
			lay = &sigmoid(*lay);

			Layer<float> &output = regulation(*lay);

			Model<float> forwardModel({&input0node}, {&output});

			SGD<float> optimizer(learningRate, 0.8);
			Adam<float> ADamoptimizer(learningRate);

			MetricAccuracy accuracy;

			forwardModel.compile(&optimizer, loss_mse, {dynamic_cast<Metric *>(&accuracy)});
			std::cout << forwardModel.summary() << std::endl;

			forwardModel.fit(epochs, inputDataXF, inputResYF, batchSize);

			forwardModel.saveWeight("mnist_forward_network_model.weight");

			Tensor<float> predict = forwardModel.predict(inputTestXF);

			/*	*/
			Tensor<float> predict_result = Tensor<float>::equal(predict, inputResTestYF);
			std::cout << predict_result << std::endl;

			// TODO Accuracy.
			std::cout << "Predict " << predict << std::endl;
		}

	} catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}