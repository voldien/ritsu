#include "layers/Regularization.h"
#include <Ritsu.h>
#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {
	const unsigned int batchSize = 1;

	Input input0node({32, 32, 1}, "input");

	Conv2D conv2D_0(32, {3, 3}, {1, 1}, "same");
	Relu relu_0;
	BatchNormalization BatchNormalization_0;

	Conv2D conv2D_1(32, {3, 3}, {1, 1}, "same");
	Relu relu_1;
	BatchNormalization BatchNormalization_1;

	Flatten flatten0("flatten0");
	Dense output(1);
	Sigmoid sigmoid;
	Regularization regularization(0.1, 0.2);

	// Layer<float> &outputLayer = regularization(sigmoid(
	//	output(flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(input0node))))))))));
	//
	// Model<float> model({&input0node}, {&outputLayer});
}