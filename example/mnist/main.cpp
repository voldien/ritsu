#include <Ritsu.h>

#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	/*	*/
	const unsigned int batchSize = 1;

	Sigmoid sigmoid;
	Relu relu;

	Tensor input({1024, 1}, 4);
	input.assignInitValue(0);

	input = input + input;
	std::cout << input << std::endl;

	Dense dense(32);

	Layer<float> *Pd = &dense;

	GuassianNoise noise(2.f, 3.f);
	input = noise(input);

	Tensor Result = relu(input);
	// Result = relu << Result;
	// Result = *Pd << input;
	std::cout << Result << std::endl;

	Input input0node({32, 32, 1}, "input");
	BatchNormalization BatchNormalization;
	Flatten flatten0("flatten0");
	Flatten flatten("flatten1");

	Dense fw0(32, true, "input0");

	Dense fw1 = Dense(128, true, "layer0");

	Dense fw2 = Dense(1, true, "layer1");

	GuassianNoise noiseLayer(0.05, 0.05f, "noise");

	Layer<float> &output = flatten(fw2(relu(fw1(noiseLayer(fw0(flatten0(input0node)))))));

	Cast<int> float2int;
	Sigmoid sig;
	Add add;
	Layer<float> &output2 = sig(add);

	Tensor inputRes({batchSize, 1}, 4);
	Tensor inputData({batchSize, 32, 32, 1}, 4);

	Model<float> forwardModel({&input0node}, {&output2});

	SGD<float> optimizer(0.0002, 0.0);

	Loss mse_loss([](const Tensor &value0, const Tensor &value1, Tensor &out) { out = value0; });
	forwardModel.compile(&optimizer, mse_loss);
	std::cout << forwardModel.summary() << std::endl;

	forwardModel.fit(1, inputData, inputRes, batchSize);
	Tensor predict = std::move(forwardModel.predict(input));

	std::cout << "Predict " << predict << std::endl;

	// Input inputRef({32, 32, 1});
	//
	// Conv2D conv(32, {2, 2}, {1, 1}, "");

	// auto &ref = conv(inputRef);

	return EXIT_SUCCESS;

	// Create N
}