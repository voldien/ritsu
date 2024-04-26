#include "layers/Dense.h"
#include "layers/GaussianNoise.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Reshape.h"
#include "layers/Tanh.h"
#include "layers/UpSampling2D.h"
#include "mnist_dataset.h"
#include <Ritsu.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {
	try {
		const unsigned int batchSize = 16;
		const unsigned int epochs = 128;
		const float learningRate = 0.0002f;
		bool useBatchNorm = false;
		bool useDropout = false;
		bool useReg = false;
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

		/*	Extract data shape.	*/
		Shape<unsigned int> dataShape = inputDataX.getShape().getSubShapeMem(1, 3);
		Shape<unsigned int> resultShape = inputResY.getShape().getSubShapeMem(1, 1);

		const Tensor<float> inputDataXF = inputDataX.cast<float>();
		const Tensor<float> inputTestXF = inputTestX.cast<float>();

		/*	*/
		std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

		Input input0node(dataShape, "input");
		Rescaling normalizedLayer(1.0f / 255.0f);
		Flatten flattenInput("flatten0");

		Dense dense_0(128, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer0");
		Relu relu_0("relu0");
		BatchNormalization BatchNormalization_0;
		Dropout drop0(0.1f);

		Dense dense_1(64, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer1");
		Relu relu_1("relu1");
		BatchNormalization BatchNormalization_1;
		Dropout drop1(0.1f);

		Dense dense_2(32, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer2");
		Relu relu_2("relu2");
		BatchNormalization BatchNormalization_2;
		Dropout drop2(0.1f);

		/*	*/
		Dense latent(8, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "latent");
		Regularization regularization(0.01, 0.02);

		/*	*/
		Dense dense_3(32, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer3");
		Relu relu_3("relu3");
		BatchNormalization BatchNormalization_3;
		Dropout drop3(0.1f);

		Dense dense_4(64, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer4");
		Relu relu_4("relu4");
		BatchNormalization BatchNormalization_4;
		Dropout drop4(0.1f);

		Dense dense_5(128, useBias, RandomNormalInitializer<float>(), RandomNormalInitializer<float>(), "layer5");
		Relu relu_5("relu5");
		BatchNormalization BatchNormalization_5;
		Dropout drop5(0.1f);

		Dense dense_6(dataShape.getNrElements(), useBias, RandomNormalInitializer<float>(),
					  RandomNormalInitializer<float>(), "layer6");
		Reshape reshape(dataShape);
		Sigmoid sigmoid;

		/*	*/
		{

			Layer<float> *lay = &normalizedLayer(input0node);
			lay = &flattenInput(*lay);

			/*	*/
			lay = &dense_0(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_0(*lay);
			}
			if (useDropout) {
				lay = &drop0(*lay);
			}
			lay = &relu_0(*lay);

			/*	*/
			lay = &dense_1(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_1(*lay);
			}
			if (useDropout) {
				lay = &drop1(*lay);
			}
			lay = &relu_1(*lay);

			/*	*/
			lay = &dense_2(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_2(*lay);
			}
			if (useDropout) {
				lay = &drop2(*lay);
			}
			lay = &relu_2(*lay);

			lay = &latent(*lay);
			if (useReg) {
				lay = &regularization(*lay);
			}
			Layer<float> *encoderLayer = lay;

			/*	*/
			lay = &dense_3(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_3(*lay);
			}
			if (useDropout) {
				lay = &drop3(*lay);
			}
			lay = &relu_3(*lay);

			/*	*/
			lay = &dense_4(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_4(*lay);
			}
			if (useDropout) {
				lay = &drop4(*lay);
			}
			lay = &relu_4(*lay);

			/*	*/
			lay = &dense_5(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_5(*lay);
			}
			if (useDropout) {
				lay = &drop5(*lay);
			}
			lay = &relu_5(*lay);

			lay = &dense_6(*lay);
			lay = &reshape(*lay);
			lay = &sigmoid(*lay);
			Layer<float> *decoderLayer = lay;

			Model<float> autoencoder({&input0node}, {decoderLayer});
			std::cout << autoencoder.summary();
			// Model<float> encoderModel({&input0node}, {encoderLayer});
			// Model<float> decoderModel({encoderLayer}, {decoderLayer});

			SGD<float> optimizer(learningRate, 0.05f);

			MetricAccuracy accuracy;
			MetricMean lossmetric("loss");
			MeanSquareError mse_loss;

			autoencoder.compile(&optimizer, mse_loss, {&accuracy});

			autoencoder.fit(epochs, inputDataXF, inputDataXF, batchSize, validationSplit);
		}
	}

	catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
