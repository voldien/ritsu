#include "layers/Add.h"
#include "layers/AveragePooling.h"
#include "layers/Cast.h"
#include "layers/Concatenate.h"
#include "layers/Conv2D.h"
#include "layers/Dense.h"
#include "layers/GaussianNoise.h"
#include "layers/Input.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Multiply.h"
#include "layers/Relu.h"
#include "layers/Sigmoid.h"
#include "layers/UpScale.h"
#include "optimizer/SGD.h"
#include <Model.h>
#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {

	Layer<float> layer("");
	Sigmoid sigmoid;
	Relu relu;

	Tensor input({1024, 1}, 4);

	input = input + input;
	std::cout << input << std::endl;

	Dense dense(32);

	Layer<float> *Pd = &dense;

	GuassianNoise noise(0.01f);
	input = noise(input);

	Tensor Result = sigmoid(relu(input));
	// Result = relu << Result;
	// Result = *Pd << input;
	std::cout << Result << std::endl;

	Input input0node({32, 32, 1});

	Dense fw0(32, true, "input0");

	Dense fw1 = Dense(128, true, "layer0");

	Dense fw2 = Dense(1, true, "layer1");

	Layer<float> &output = sigmoid(relu(fw2(relu(fw1(fw0(input0node))))));

	Cast<int> float2int;
	Sigmoid sig;
	Add add(output, fw2);
	Layer<float> output2 = sig(add);

	Tensor inputRes({1, 1}, 4);

	Model<float> forwardModel({&fw0}, {&output2});

	SGD<float> optimizer(0.0001, 0.9);
	Loss loss;
	forwardModel.compile(&optimizer, loss);
	forwardModel.fit(1, input, inputRes);
	Tensor predict = std::move(forwardModel.predict(input));

	std::cout << "Predict " << predict << std::endl;

	//Input inputRef({32, 32, 1});
//
	//Conv2D conv(32, {2, 2}, {1, 1}, "");

	// auto &ref = conv(inputRef);

	return EXIT_SUCCESS;

	// Create N
}