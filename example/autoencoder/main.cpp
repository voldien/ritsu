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
		const unsigned int batchSize = 1;
		const unsigned int epochs = 128;
		const float learningRate = 0.0002f;
		bool useBatchNorm = false;

		Shape<unsigned int> dataShape({32, 32, 1});
		Shape<unsigned int> resultShape({10});

		/*	*/
		Tensor<uint8_t> inputResY;
		Tensor<uint8_t> inputResTestY;

		Tensor<uint8_t> inputDataX;
		Tensor<uint8_t> inputTestX;

		/*	*/
		RitsuDataSet::loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
								"t10k-labels.idx1-ubyte", inputDataX, inputResY, inputTestX, inputResTestY);

		Input input0node(dataShape, "input");
		Cast<uint8_t, float> cast2Float;
		Rescaling normalizedLayer(1.0f / 255.0f);

		Conv2D conv2D_0(32, {3, 3}, {2, 2}, ConvPadding::Same);
		Relu relu_0;
		BatchNormalization BatchNormalization_0;
		Add encAdd0;

		Conv2D conv2D_1(64, {3, 3}, {1, 1}, ConvPadding::Same);
		Relu relu_1;
		BatchNormalization BatchNormalization_1;

		Conv2D conv2D_2(64, {3, 3}, {1, 1}, ConvPadding::Same);
		Relu relu_2;
		BatchNormalization BatchNormalization_2;

		Flatten flatten0("flatten0");
		Regularization regularization(0.1, 0.2);
		Reshape reshape(Shape<unsigned int>({256, 256, 32}));

		UpSampling2D<float> upscale0(2);
		Conv2D conv2D_3(32, {3, 3}, {2, 2}, ConvPadding::Same);
		Relu relu_3;
		BatchNormalization BatchNormalization_3;

		UpSampling2D<float> upscale1(2);
		Conv2D conv2D_4(64, {3, 3}, {1, 1}, ConvPadding::Same);
		Relu relu_4;
		BatchNormalization BatchNormalization_4;

		UpSampling2D<float> upscale3(2);
		Conv2D conv2D_5(64, {3, 3}, {1, 1}, ConvPadding::Same);
		Relu relu_5;
		BatchNormalization BatchNormalization_5;

		/*	*/
		{
			Layer<float> *lay = &normalizedLayer(input0node);

			// lay = &noise(*lay);

			lay = &flatten0(*lay);

			/*	*/
			// Layer<float> &encoder =
			//	output(flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(input0node))))))));
			//
			// Layer<float> &decoder =
			//	output(flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(encoder))))))));

			SGD<float> optimizer(learningRate, 0.0);

			MetricAccuracy accuracy;
			MetricMean lossmetric("loss");
			//
			// Model<float> model({&input0node}, {&decoder});

			// Tensor<float> predict = std::move(model.predict(inputTestX));
			// Compare.
			// std::cout << "Predict " << predict << std::endl;
		}
	} catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}