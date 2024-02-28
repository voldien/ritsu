#include "layers/Conv2D.h"
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
	Tensor<uint8_t> inputResY;
	Tensor<uint8_t> inputResTestY;

	Tensor<uint8_t> inputDataX;
	Tensor<uint8_t> inputTestX;

	/*	*/
	RitsuDataSet::loadMNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
							"t10k-labels.idx1-ubyte", inputDataX, inputResY, inputTestX, inputResTestY);

	std::cout << "Loaded MNIST Data Set: " << inputDataX.getShape() << " Labels: " << inputResY.getShape() << std::endl;

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

	bool useResnet;
	bool useBatchNorm = false;

	/*	*/
	{
		Input input0node(dataShape, "input");
		Rescaling normalizedLayer(1.0f / 255.0f);

		Conv2D conv2D_0(32, {3, 3}, {2, 2}, ConvPadding::Same);
		Relu relu_0;
		BatchNormalization BatchNormalization_0;

		Conv2D conv2D_1(64, {3, 3}, {2, 2}, ConvPadding::Same);
		Relu relu_1;
		BatchNormalization BatchNormalization_1;

		Conv2D conv2D_2(128, {3, 3}, {2, 2}, ConvPadding::Same);
		Relu relu_2;
		BatchNormalization BatchNormalization_2;

		Flatten flatten0("flatten0");
		Dense output(output_size);
		
		Sigmoid sigmoid;
		Regularization regularization(0.1f, 0.2f);

		/*	*/
		{
			// Layer<float> *lay = &cast2Float(input0node);

			// lay = &normalizedLayer(*lay);
			Layer<float> *lay = &normalizedLayer(input0node);

			lay = &conv2D_0(*lay);
			lay = &relu_0(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_0(*lay);
			}

			lay = &conv2D_1(*lay);
			lay = &relu_1(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_1(*lay);
			}

			lay = &conv2D_2(*lay);
			lay = &relu_2(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_2(*lay);
			}

			lay = &flatten0(*lay);
			lay = &output(*lay);
			lay = &sigmoid(*lay);

			Layer<float> &output = regularization(*lay);

			Layer<float> &outputLayer = regularization(sigmoid(output(
				flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(input0node))))))))));

			Model<float> forwardModel({&input0node}, {&output});

			SGD<float> optimizer(learningRate, 0.0);
			MetricAccuracy accuracy;
			MetricMean lossmetric("loss");

			Loss mse_loss(loss_mse);
			forwardModel.compile(&optimizer, loss_error,
								 {dynamic_cast<Metric *>(&lossmetric), dynamic_cast<Metric *>(&accuracy)});
			std::cout << forwardModel.summary() << std::endl;

			forwardModel.fit(epochs, inputDataXF, inputResYF, batchSize);

			Tensor<float> predict = forwardModel.predict(inputTestXF);

			/*	*/
			Tensor<float> predict_result = Tensor<float>::equal(predict, inputResTestYF);
			std::cout << predict_result << std::endl;

			// TODO Accuracy.
			std::cout << "Predict " << predict << std::endl;
		}
	}
}