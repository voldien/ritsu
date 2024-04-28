#include "Loss.h"
#include "Metric.h"
#include "Tensor.h"
#include "layers/Dropout.h"
#include "layers/GaussianNoise.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Relu.h"
#include "layers/Rescaling.h"
#include "layers/Sigmoid.h"
#include "layers/SoftMax.h"
#include "mnist_dataset.h"
#include <Ritsu.h>
#include <cstdint>
#include <exception>
#include <iostream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	try {
		/*	*/
		const unsigned int batchSize = 64;
		const unsigned int epochs = 128;
		const float learningRate = 0.000001f;
		bool useBatchNorm = false;
		bool useSigmoidAct = true;
		bool useDropout = true;
		const float validationSplit = 0.1f;
		bool useBias = false;
		bool useNoise = false;

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

		GuassianNoise noise(0, 0.8f);

		Flatten flattenInput("flatten0");
		Flatten flatten("flatten1");

		Dense fw0(32, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer0");
		BatchNormalization BN0;
		Dropout drop0(0.3f);
		Relu relu0("relu0");

		Dense fw1 = Dense(16, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer1");
		BatchNormalization BN1;
		Dropout drop1(0.3f);
		Relu relu1("relu1");

		Dense fw2_output =
			Dense(output_size, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer2");

		Regularization regulation(0.00001f, 0.000f);

		Sigmoid sigmoid;
		SoftMax outputAct;

		/*	*/
		{
			Layer<float> *lay = &normalizedLayer(input0node);
			lay = &flattenInput(*lay);

			if (useNoise) {
				lay = &noise(*lay);
			}

			lay = &fw0(*lay);
			if (useBatchNorm) {
				lay = &BN0(*lay);
			}

			if (useDropout) {
				lay = &drop0(*lay);
			}

			lay = &relu0(*lay);

			lay = &fw1(*lay);
			if (useBatchNorm) {
				lay = &BN1(*lay);
			}
			if (useDropout) {
				lay = &drop1(*lay);
			}
			lay = &relu1(*lay);

			lay = &fw2_output(*lay);
			lay = &sigmoid(*lay);

			Layer<float> &output = regulation(*lay);

			Model<float> forwardModel({&input0node}, {&output});

			SGD<float> optimizer(learningRate, 0.0008f);

			MetricAccuracy accuracy;
			CategoricalCrossentropy cross_loss(true);
			MeanSquareError mse_loss;
			forwardModel.compile(&optimizer, mse_loss, {&accuracy});
			std::cout << forwardModel.summary() << std::endl;

			forwardModel.fit(epochs, inputDataXF, inputResYF, batchSize, validationSplit);

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