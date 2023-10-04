#include "Metric.h"
#include "Tensor.h"
#include "layers/Regularization.h"
#include "layers/UpScale.h"
#include <Ritsu.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

void loadMNIST(const std::string &imagePath, const std::string &labelPath, Tensor &dataX, Tensor &dataY) {
	/*	*/
	std::ifstream imageStream(imagePath, std::ios::in | std::ios::binary);
	std::ifstream labelStream(labelPath, std::ios::in | std::ios::binary);

	if (imageStream.is_open()) {

		/*	*/
		int32_t width, height, nr_images, image_magic;

		imageStream.seekg(0, std::ios::beg);

		imageStream.read((char *)&image_magic, sizeof(image_magic));
		imageStream.read((char *)&nr_images, sizeof(nr_images));
		imageStream.read((char *)&width, sizeof(width));
		imageStream.read((char *)&height, sizeof(height));

		const size_t ImageSize = static_cast<size_t>(width) * static_cast<size_t>(height);

		dataX = Tensor({nr_images, width, height, 1}, sizeof(uint8_t));
		uint8_t *raw = dataX.getRawData<uint8_t>();

		uint8_t *imageData = (uint8_t *)malloc(ImageSize);

		for (size_t i = 0; i < nr_images; i++) {
			imageStream.read((char *)&imageData[0], ImageSize);
			memcpy(&raw[i * ImageSize], imageData, ImageSize);
		}

		free(imageData);
	}

	uint32_t label_magic, nr_label;

	labelStream.seekg(0, std::ios::beg);
	labelStream.read((char *)&label_magic, sizeof(label_magic));
	labelStream.read((char *)&nr_label, sizeof(nr_label));

	dataY = Tensor({nr_label}, sizeof(uint32_t));
	uint32_t label;
	for (size_t i = 0; i < nr_label; i++) {

		labelStream.read((char *)&label, sizeof(label));
		dataX.getValue<uint32_t>(i) = label;
	}
}

int main(int argc, const char **argv) {

	/*	*/
	const unsigned int batchSize = 1;
	const unsigned int output_size = 10;
	const unsigned int epochs = 128;
	Shape<unsigned int> dataShape({32, 32, 1});
	Shape<unsigned int> resultShape({10});

	const size_t dataBufferSize = 5;
	Tensor inputResY({dataBufferSize, output_size}, 4);
	Tensor inputDataX({dataBufferSize, 32, 32, 1}, 4);

	loadMNIST("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", inputDataX, inputResY);

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

	Regularization regulation(0.00, 0.001);

	Sigmoid outputAct;

	Layer<float> &output = regulation(outputAct(fw2(relu1(BN1(fw1(relu0(BN0(fw0(flattenInput(input0node))))))))));

	Model<float> forwardModel({&input0node}, {&output});

	SGD<float> optimizer(0.002f, 0.0);

	MetricAccuracy accuracy;
	MetricMean lossmetric("loss");

	Loss mse_loss(loss_mse);
	forwardModel.compile(&optimizer, loss_mse, {dynamic_cast<Metric *>(&lossmetric), (Metric *)&accuracy});
	std::cout << forwardModel.summary() << std::endl;

	forwardModel.fit(epochs, inputDataX, inputResY, batchSize);
	Tensor predict = std::move(forwardModel.predict(inputDataX));
	std::cout << "Predict " << predict << std::endl;

	return EXIT_SUCCESS;
}