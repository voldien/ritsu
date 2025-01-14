#include "Loss.h"
#include "Metric.h"
#include "Tensor.h"
#include "layers/Dropout.h"
#include "layers/GaussianNoise.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Relu.h"
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
		"b,use-bias", "Use Dense Bias", cxxopts::value<bool>()->default_value("true"))(
		"m,mid-dense-count", "Set Number of neuron in middle layer", cxxopts::value<int>()->default_value("2"))(
		"l,learning-rate", "Set Learning Rate", cxxopts::value<float>()->default_value("0.00001"))(
		"M,optimizer-momentum", "Set Optimizer momentum", cxxopts::value<float>()->default_value("0.1"))(
		"V,validation", "Set Validation split", cxxopts::value<float>()->default_value("0.1"))(
		"S,seed", "Set Seed", cxxopts::value<int>()->default_value("1234"))(
		"T,trainig-size", "Set Training Size", cxxopts::value<size_t>()->default_value("65536"))(
		"O,optimizer", "Set Optimizer ", cxxopts::value<std::string>()->default_value("sgd"))(
		"s,use-sigmoid", " ", cxxopts::value<bool>()->default_value("false"))(
		"L,loss-funciton", " ", cxxopts::value<std::string>()->default_value("mse"))(
		"r,regulation", " ", cxxopts::value<float>()->default_value("0.00000"))(
		"t,threads", "Set number of threads (Core) to use", cxxopts::value<int>()->default_value("-1"));

	/*	Parse the command line input.	*/
	auto result = options.parse(argc, (char **&)argv);

	/*	*/
	const bool debug = result["debug"].as<bool>();
	const unsigned int batchSize = result["batch"].as<int>();
	const unsigned int epochs = result["epoch"].as<int>();
	const float learningRate = result["learning-rate"].as<float>();
	const bool useBatchNorm = false;
	const bool useSigmoidAct = result["use-sigmoid"].as<bool>();
	const bool useDropout = false;
	const float validationSplit = Math::clamp<float>(result["validation"].as<float>(), 0, 1);
	const bool useBias = result["use-bias"].as<bool>();
	const bool useNoise = result["use-noise"].as<bool>();
	const float momentum = result["optimizer-momentum"].as<float>();
	const float useRegulation = result["regulation"].as<float>();
	const int threads = result["threads"].as<int>();

	const std::string use_optimizer = result["optimizer"].as<std::string>();
	const std::string use_loss_function = result["loss-funciton"].as<std::string>();

	if (threads == -1) {
		omp_set_num_threads(omp_get_num_procs());
	} else {
		omp_set_num_threads(threads);
	}

	if (debug) {
		/*	*/
		Ritsu::enableDebug();
	}

	/*	*/
	srand(time(nullptr));

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
		const Tensor<float> inputResYF = std::move(inputResY.cast<float>() * (1.0 / 255.0f));
		const Tensor<float> inputResTestYF = std::move(inputResTestY.cast<float>() * (1.0 / 255.0f));

		/*	Extract data shape.	*/
		Shape<unsigned int> dataShape = inputDataX.getShape().getSubShapeMem(1, 3);
		Shape<unsigned int> resultShape = inputResYF.getShape().getSubShapeMem(1, 1);
		const unsigned int output_size = 10;

		const Tensor<float> inputDataXF = std::move(inputDataX.cast<float>() * (1.0 / 255.0f));
		const Tensor<float> inputTestXF = std::move(inputTestX.cast<float>() * (1.0 / 255.0f));

		/*	*/
		std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

		/*	Creat all layers.	*/
		Input input0node(dataShape, "input");
		Cast<uint8_t, float> cast2Float;

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

		Regularization regulation(useRegulation, 0.000f);

		Sigmoid sigmoid;
		SoftMax outputAct;

		/*	Connect layers.	*/
		{
			Layer<float> *lay = &flattenInput(input0node);

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

			if (useRegulation > 0) {
				lay = &regulation(*lay);
			}

			Layer<float> &output = *lay;

			Model<float> forwardModel({&input0node}, {&output});

			SGD<float> optimizer(learningRate, momentum);
			Adam<float> Adamoptimizer(learningRate);

			MetricAccuracy accuracy;

			/*	*/
			Loss<float> *lossfunction = nullptr;
			CategoricalCrossentropy cross_loss(true);
			MeanSquareError mse_loss;

			/*	*/
			if (useSigmoidAct) {
				lossfunction = &mse_loss;
			} else {
				lossfunction = &cross_loss;
			}

			/*	*/
			forwardModel.compile(&optimizer, *lossfunction, {&accuracy});
			std::cout << forwardModel.summary() << std::endl;

			/*	*/
			Model<float>::History &history =
				forwardModel.fit(epochs, inputDataXF, inputResYF, batchSize, validationSplit);

			/*	*/
			forwardModel.saveWeight("mnist_forward_network_model.weight");

			/*	Calculate test Loss.	*/
			Tensor<float> predict = forwardModel.predict<float, float>(inputTestXF);
			Tensor<float> predicted_loss = lossfunction->computeLoss(inputResTestYF, predict);
			MetricAccuracy test_accuracy;
			test_accuracy.update_state({&predict, &inputResTestYF});

			/*	*/
			std::cout << std::endl << "Average Test Loss: " << predicted_loss.mean() << std::endl;
			std::cout << std::endl << "Accuracy Test: " << test_accuracy.result() << std::endl;
		}

	} catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
