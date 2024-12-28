#include "Metric.h"
#include "Tensor.h"
#include "mnist_dataset.h"
#include "optimizer/Optimizer.h"
#include <Ritsu.h>
#include <cstdio>
#include <cxxopts.hpp>
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
		"O,optimizer", "Set Optimizer ", cxxopts::value<std::string>()->default_value("sgd"));

	/*	Parse the command line input.	*/
	auto result = options.parse(argc, (char **&)argv);

	/*	*/
	const bool debug = result["debug"].as<bool>();
	const unsigned int batchSize = result["batch"].as<int>();
	const unsigned int epochs = result["epoch"].as<int>();
	const float learningRate = result["learning-rate"].as<float>();
	const size_t trainingDataSize = result["trainig-size"].as<size_t>();
	const float validationSplit = Math::clamp<float>(result["validation"].as<float>(), 0, 1);
	const bool useBias = result["use-bias"].as<bool>();
	const bool useNoise = result["use-noise"].as<bool>();
	const unsigned int dense_size = result["mid-dense-count"].as<int>();
	const float momentum = result["optimizer-momentum"].as<float>();

	/*	*/
	const unsigned int input_size = 2;
	const unsigned int output_size = 1;

	try {

		/*	Create dataset.	*/
		const size_t random_seed = 10052;
		RandomNormalInitializer<float> random(0, 3, random_seed);
		Tensor<float> dataX = random(Shape<unsigned int>({static_cast<unsigned int>(trainingDataSize), input_size}));
		Tensor<float> testX = random(Shape<unsigned int>({1, input_size}));
		/*	*/
		Tensor<float> dataY({static_cast<unsigned int>(trainingDataSize), 1});
		for (unsigned int index = 0; index < dataY.getNrElements(); index++) {
			const float value = dataX.getValue({index, 0}) + dataX.getValue({index, 1});
			dataY.getValue(index) = value;
		}

		/*	Create layers.	*/
		Input input({input_size}, "input");
		Dense dense0(dense_size, useBias);
		Dense outputDense(output_size, useBias);

		/*	Connect layers.	*/
		Layer<float> &output = outputDense(dense0(input));

		/*	Setup optimizer.	*/
		Optimizer<float> *optimizer = nullptr;
		SGD<float> optimizerSGD(learningRate, momentum);
		Adam<float> Adamoptimizer(learningRate);

		MetricAccuracy accuracy;
		Model<float> forwardModel = Model<float>({&input}, {&output});
		const Loss<float> &mse_loss = MeanSquareError();

		forwardModel.compile(&Adamoptimizer, mse_loss, {dynamic_cast<Metric *>(&accuracy)});
		std::cout << forwardModel.summary() << std::endl;

		Model<float>::History *result = &forwardModel.fit(epochs, dataX, dataY, batchSize, validationSplit, true, true);

		forwardModel.saveWeight("mnist_forward_network_model.weight");

		const Tensor<float> predict = forwardModel.predict<float, float>(testX);

		/*	Ouput final result.	*/
		{
			std::cout << "Input: " << testX << std::endl;
			std::cout << "Predict: " << predict << std::endl;
		}

	} catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}