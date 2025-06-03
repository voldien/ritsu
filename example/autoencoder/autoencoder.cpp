#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Reshape.h"
#include "mnist_dataset.h"
#include <Ritsu.h>
#include <cstdlib>
#include <cxxopts.hpp>
#include <iostream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	/*	Parse argument.	*/
	cxxopts::Options options("Auto Encoder Neural Network");
	cxxopts::OptionAdder &addr = options.add_options("Addition")("h,help", "helper information.")(
		"d,debug", "Enable Debugging", cxxopts::value<bool>()->default_value("false"))(
		"B,batch", "Set the Batch Size", cxxopts::value<int>()->default_value("1"))(
		"N,use-noise", "Enable the use of noise", cxxopts::value<bool>()->default_value("false"))(
		"E,epoch", "Set the number of epochs", cxxopts::value<int>()->default_value("8"))(
		"b,use-bias", "Use Dense Bias", cxxopts::value<bool>()->default_value("true"))(
		"m,mid-dense-count", "Set Number of neuron in middle layer", cxxopts::value<int>()->default_value("2"))(
		"l,learning-rate", "Set Learning Rate", cxxopts::value<float>()->default_value("0.00000001"))(
		"M,optimizer-momentum", "Set Optimizer momentum", cxxopts::value<float>()->default_value("0.0"))(
		"V,validation", "Set Validation split", cxxopts::value<float>()->default_value("0.1"))(
		"S,seed", "Set Seed", cxxopts::value<int>()->default_value("1234"))(
		"T,trainig-size", "Set Training Size", cxxopts::value<size_t>()->default_value("65536"))(
		"O,optimizer", "Set Optimizer ", cxxopts::value<std::string>()->default_value("sgd"))(
		"L,loss-funciton", " ", cxxopts::value<std::string>()->default_value("mse"))(
		"r,regulation", " ", cxxopts::value<float>()->default_value("0.00000"))(
		"t,threads", "Set number of threads (Core) to use", cxxopts::value<int>()->default_value("-1"));

	/*	Parse the command line input.	*/
	auto result = options.parse(argc, (char **&)argv);

	try {
		const bool debug = result["debug"].as<bool>();
		const unsigned int batchSize = result["batch"].as<int>();
		const unsigned int epochs = result["epoch"].as<int>();
		const float learningRate = result["learning-rate"].as<float>();
		bool useBatchNorm = false;
		bool useDropout = false;
		const float validationSplit = Math::clamp<float>(result["validation"].as<float>(), 0, 1);
		const bool useBias = result["use-bias"].as<bool>();
		const bool useNoise = result["use-noise"].as<bool>();
		const float momentum = result["optimizer-momentum"].as<float>();
		const float useRegulation = result["regulation"].as<float>();
		const int threads = result["threads"].as<int>();

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

		/*	Release unused dataset.	*/
		inputResY.release();
		inputResTestY.release();

		/*	Extract data shape.	*/
		Shape<unsigned int> dataShape = inputDataX.getShape().getSubShapeMem(1, 3);
		Shape<unsigned int> resultShape = dataShape;

		const Tensor<float> inputDataXF = inputDataX.cast<float>() * (1.0f / 255.0f);
		const Tensor<float> inputTestXF = inputTestX.cast<float>() * (1.0f / 255.0f);

		/*	*/
		std::cout << "Train Object Size: " << dataShape << " Expected result Size: " << resultShape << std::endl;

		Input input0node(dataShape, "input");
		Flatten flattenInput("flatten0");

		Dense dense_0(128, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer0");
		Relu relu_0("relu0");
		BatchNormalization BatchNormalization_0;
		Dropout drop0(0.1f);

		Dense dense_1(64, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer1");
		Relu relu_1("relu1");
		BatchNormalization BatchNormalization_1;
		Dropout drop1(0.1f);

		Dense dense_2(32, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer2");
		Relu relu_2("relu2");
		BatchNormalization BatchNormalization_2;
		Dropout drop2(0.1f);

		/*	*/
		Dense latent(8, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "latent");
		Regularization regularization(useRegulation, 0);

		/*	*/
		Dense dense_3(32, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer3");
		Relu relu_3("relu3");
		BatchNormalization BatchNormalization_3;
		Dropout drop3(0.1f);

		Dense dense_4(64, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer4");
		Relu relu_4("relu4");
		BatchNormalization BatchNormalization_4;
		Dropout drop4(0.1f);

		Dense dense_5(128, useBias, RandomUniformInitializer<float>(), RandomUniformInitializer<float>(), "layer5");
		Relu relu_5("relu5");
		BatchNormalization BatchNormalization_5;
		Dropout drop5(0.1f);

		Dense dense_6(dataShape.getNrElements(), useBias, RandomUniformInitializer<float>(),
					  RandomUniformInitializer<float>(), "layer6");
		Reshape reshape(dataShape);
		Sigmoid sigmoid;

		/*	*/
		{

			Layer<float> *lay = &flattenInput(input0node);

			/*	Layer0	*/
			lay = &dense_0(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_0(*lay);
			}
			if (useDropout) {
				lay = &drop0(*lay);
			}
			lay = &relu_0(*lay);

			/*	Layer1	*/
			lay = &dense_1(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_1(*lay);
			}
			if (useDropout) {
				lay = &drop1(*lay);
			}
			lay = &relu_1(*lay);

			/*	Layer2	*/
			lay = &dense_2(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_2(*lay);
			}
			if (useDropout) {
				lay = &drop2(*lay);
			}
			lay = &relu_2(*lay);

			/*	Latent Space Layer*/
			lay = &latent(*lay);
			if (useRegulation > 0) {
				lay = &regularization(*lay);
			}
			Layer<float> *encoderLayer = lay;

			/*	Layer3	*/
			lay = &dense_3(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_3(*lay);
			}
			if (useDropout) {
				lay = &drop3(*lay);
			}
			lay = &relu_3(*lay);

			/*	Layer4	*/
			lay = &dense_4(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_4(*lay);
			}
			if (useDropout) {
				lay = &drop4(*lay);
			}
			lay = &relu_4(*lay);

			/*	Layer5	*/
			lay = &dense_5(*lay);
			if (useBatchNorm) {
				lay = &BatchNormalization_5(*lay);
			}
			if (useDropout) {
				lay = &drop5(*lay);
			}
			lay = &relu_5(*lay);

			/*	Output layer.	*/
			lay = &dense_6(*lay);
			lay = &sigmoid(*lay);
			lay = &reshape(*lay);
			Layer<float> *decoderLayer = lay;

			Model<float> autoencoder({&input0node}, {decoderLayer});

			Model<float> encoderModel({&input0node}, {encoderLayer});
			Model<float> decoderModel({encoderLayer}, {decoderLayer});

			Optimizer<float> *optimizer = nullptr;
			SGD<float> optimizerSGD(learningRate, momentum);
			Adam<float> AdamOptimizer(learningRate);

			MetricAccuracy accuracy;
			MetricMean lossmetric("loss");
			MeanSquareError mse_loss;

			autoencoder.compile(&AdamOptimizer, mse_loss, {&accuracy});

			std::cout << autoencoder.summary();

			autoencoder.fit<float,float>(epochs, inputDataXF, inputDataXF, batchSize, validationSplit);

			encoderModel.saveWeight("autoencoder_network_model.weight");
			decoderModel.saveWeight("autoencoder_network_model.weight");

			Tensor<float> predicted = autoencoder.predict<float, float>(inputTestXF);
			/*	*/
			Tensor<float> testLoss = mse_loss.computeLoss(inputTestXF, predicted);

			MetricAccuracy test_accuracy;
			test_accuracy.update_state({&predicted, &inputTestXF});

			/*	*/
			std::cout << "Average Test Loss: " << testLoss.mean() << std::endl;
			std::cout << "Accuracy Test: " << test_accuracy.result() << std::endl;
		}
	}

	catch (std::exception &ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
