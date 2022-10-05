#include "layers/Dense.h"
#include "layers/Layer.h"
#include <stdio.h>

using namespace Ritsu;

int main(int argc, const char **argv) {

	Layer<float> layer("");

	Tensor input({32, 1}, 4);

	Dense dense(32);

	Layer<float> *Pd = &dense;

	Tensor Result = dense << input;
	Result = *Pd << input;

	return EXIT_SUCCESS;
}