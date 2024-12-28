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
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	/*	Parse argument.	*/
	cxxopts::Options options("Addition Neural Network");
	cxxopts::OptionAdder &addr = options.add_options("Addition")("h,help", "helper information.")(
		"d,debug", "Enable Debugging", cxxopts::value<bool>()->default_value("false"))(
		"B,batch", "Set the Batch Size", cxxopts::value<int>()->default_value("1"))(
		"N,use-noise", "Enable the use of noise", cxxopts::value<bool>()->default_value("false"))(
		"E,epoch", "Set the number of epochs", cxxopts::value<int>()->default_value("8"))(
		"b,use-bias", "Use Dense Bias", cxxopts::value<bool>()->default_value("false"))(
		"m,mid-dense-count", "Set Number of neuron in middle layer", cxxopts::value<int>()->default_value("2"))(
		"l,learning-rate", "Set Learning Rate", cxxopts::value<float>()->default_value("0.00000001"))(
		"M,optimizer-momentum", "Set Optimizer momentum", cxxopts::value<float>()->default_value("0.1"))(
		"V,validation", "Set Validation split", cxxopts::value<float>()->default_value("0.1"))(
		"S,seed", "Set Seed", cxxopts::value<int>()->default_value("1234"))(
		"T,trainig-size", "Set Training Size", cxxopts::value<size_t>()->default_value("65536"))(
		"O,optimizer", "Set Optimizer ", cxxopts::value<std::string>()->default_value("sgd"))
		(
		"s,use-sigmoid", " ", cxxopts::value<bool>()->default_value("false"));

	/*	Parse the command line input.	*/
	auto result = options.parse(argc, (char **&)argv);

	/*	*/
	const bool debug = result["debug"].as<bool>();
	const unsigned int batchSize = 2; // result["batch"].as<int>();
	const unsigned int epochs = result["epoch"].as<int>();
	const float learningRate = result["learning-rate"].as<float>();
	const bool useBatchNorm = false;
	const bool useSigmoidAct = result["use-sigmoid"].as<bool>();
	const bool useDropout = false;
	const float validationSplit = Math::clamp<float>(result["validation"].as<float>(), 0, 1);
	const bool useBias = result["use-bias"].as<bool>();
	const bool useNoise = result["use-noise"].as<bool>();
	const float momentum = result["optimizer-momentum"].as<float>();
	const bool useRegulation = false;

	if (debug) {
		/*	*/
		Ritsu::enableDebug();
	}

	try {

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
		inputResY = std::move(Tensor<uint8_t>::oneShot(inputResY));
		inputResTestY = std::move(Tensor<uint8_t>::oneShot(inputResTestY));

		/*	*/
		const Tensor<float> inputResYF = std::move(inputResY.cast<float>());
		const Tensor<float> inputResTestYF = std::move(inputResTestY.cast<float>());

		/*	Extract data shape.	*/
		Shape<unsigned int> dataShape = inputDataX.getShape().getSubShapeMem(1, 3);
		Shape<unsigned int> resultShape = inputResYF.getShape().getSubShapeMem(1, 1);
		const unsigned int output_size = 10;

		const Tensor<float> inputDataXF = std::move(inputDataX.cast<float>());
		const Tensor<float> inputTestXF = std::move(inputTestX.cast<float>());

		/*	*/
		std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

		/*	Creat all layers.	*/
		Input input0node(dataShape, "input");
		Cast<uint8_t, float> cast2Float;
		Rescaling normalizedLayer(1.0f / 255.0f);

		GuassianNoise noise(0, 0.8f);

		Flatten flattenInput("flatten0");
		Flatten flatten("flatten1");

		Dense fw0(32, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer0");
		BatchNormalization BN0;
		Dropout drop0(0.3f);
		Relu relu0("relu0");

		Dense fw1 = Dense(16, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer1");
		BatchNormalization BN1;
		Dropout drop1(0.3f);
		Relu relu1("relu1");

		Dense fw2_output =
			Dense(output_size, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer2");

		Regularization regulation(0.00001f, 0.000f);

		Sigmoid sigmoid;
		SoftMax outputAct;

		/*	Connect layers.	*/
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

			if (useSigmoidAct) {
				lay = &sigmoid(*lay);
			}

			if (useRegulation) {
				lay = &regulation(*lay);
			}

			Layer<float> &output = *lay;

			Model<float> forwardModel({&input0node}, {&output});

			SGD<float> optimizer(learningRate, momentum);
			Adam<float> Adamoptimizer(learningRate);

			MetricAccuracy accuracy;

			Loss<float> *lossfunction = nullptr;
			CategoricalCrossentropy cross_loss(true);
			MeanSquareError mse_loss;
			if (useSigmoidAct) {
				lossfunction = &mse_loss;
			} else {
				lossfunction = &cross_loss;
			}

			forwardModel.compile(&Adamoptimizer, *lossfunction, {&accuracy});
			std::cout << forwardModel.summary() << std::endl;

			forwardModel.fit(epochs, inputDataXF, inputResYF, batchSize, validationSplit);

			forwardModel.saveWeight("mnist_forward_network_model.weight");

			Tensor<float> predict = forwardModel.predict<float, float>(inputTestXF);

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