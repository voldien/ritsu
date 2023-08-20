#include "layers/GaussianNoise.h"
#include "layers/Layer.h"
#include "layers/Regularization.h"
#include "layers/Reshape.h"
#include "layers/Tanh.h"
#include <Ritsu.h>
#include <cstdio>
#include <iostream>
#include <istream>
#include <ostream>

using namespace Ritsu;

int main(int argc, const char **argv) {
	const unsigned int batchSize = 1;

	Input input0node({32, 32, 1}, "input");

	GuassianNoise noiseLayer(0.2, 0.3f);

	Conv2D conv2D_0(32, {3, 3}, {1, 1}, "same");
	Relu relu_0;
	BatchNormalization BatchNormalization_0;

	Conv2D conv2D_1(32, {3, 3}, {1, 1}, "same");
	Relu relu_1;
	BatchNormalization BatchNormalization_1;

	Flatten flatten0("flatten0");
	Sigmoid sigmoid;
	Regularization regularization(0.1, 0.2);
	Tahn output;
	Reshape reshape(Shape<unsigned int>({256, 256, 32}));

	/*	*/
	Layer<float> &encoder =
		output(flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(input0node))))))));

	Layer<float> &decoder =
		output(flatten0(relu_0(BatchNormalization_1(conv2D_1(relu_0(BatchNormalization_0(conv2D_0(encoder))))))));

	//
	Model<float> model({&input0node}, {&decoder});
}