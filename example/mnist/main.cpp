#include <Ritsu.h>

#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

void loadMNIST(const std::string& imagePath ){
	
}

int main(int argc, const char **argv) {

	/*	*/
	const unsigned int batchSize = 1;
	const unsigned int epochs = 128;

	Sigmoid sigmoid;
	Relu relu;

	Tensor input({1024, 1}, 4);
	input.assignInitValue(0);

	input = input + input;
	// std::cout << input << std::endl;

	Dense dense(32);

	Layer<float> *Pd = &dense;

	GuassianNoise noise(2.f, 3.f);
	input = noise(input);

	Tensor Result = relu(input);
	// Result = relu << Result;
	// Result = *Pd << input;
	// std::cout << Result << std::endl;

	Input input0node({32, 32, 1}, "input");
	BatchNormalization BatchNormalization;
	Flatten flatten0("flatten0");
	Flatten flatten("flatten1");

	Dense fw0(128, true, "layer0");

	Dense fw1 = Dense(128, true, "layer1");

	Dense fw2 = Dense(10, true, "layer2");

	Sigmoid outputAct;

	GuassianNoise noiseLayer(0.05, 0.05f, "noise");

	Layer<float> &output = outputAct(flatten(fw2(relu(fw1(noiseLayer(fw0(flatten0(input0node))))))));

	Cast<int> float2int;
	Sigmoid sig;
	Add add;
	Layer<float> &output2 = sig(add);

	const size_t dataBufferSize = 100;
	Tensor inputRes({dataBufferSize, 10}, 4);
	Tensor inputData({dataBufferSize, 32, 32, 1}, 4);

	Model<float> forwardModel({&input0node}, {&output2});

	SGD<float> optimizer(0.002, 0.0);

	Loss mse_loss(loss_mse);
	forwardModel.compile(&optimizer, loss_cross_entropy);
	std::cout << forwardModel.summary() << std::endl;

	forwardModel.fit(epochs, inputData, inputRes, batchSize);
	Tensor predict = std::move(forwardModel.predict(input));
	std::cout << "Predict " << predict << std::endl;

	// Input inputRef({32, 32, 1});
	//
	// Conv2D conv(32, {2, 2}, {1, 1}, "");

	// auto &ref = conv(inputRef);

	return EXIT_SUCCESS;

	// Create N
}