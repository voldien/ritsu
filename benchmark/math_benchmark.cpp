#include "Activations.h"
#include "layers/Relu.h"
#include "layers/Sigmoid.h"
#include "layers/SoftMax.h"
#include <Ritsu.h>
#include <benchmark/benchmark.h>

static void BM_MathProduct(benchmark::State &state) {
	// Perform setup here
	std::vector<float> x(4096, 1);
	volatile auto sum = 0.0f;
	for (auto _ : state) {
		sum = Ritsu::Math::product(x);
	}
	assert(sum == 4096);
}

static void BM_MathSum(benchmark::State &state) {
	// Perform setup here
	std::vector<float> x(4096, 1);
	volatile auto sum = 0.0f;
	for (auto _ : state) {
		sum = Ritsu::Math::sum(x);
	}
	assert(sum == 1);
}

static void BM_MathRelu(benchmark::State &state) {
	// Perform setup here
	std::vector<float> x(512 * 512, 1);
	volatile auto sum = 0.0f;
	for (auto _ : state) {
		unsigned int index;
#pragma omp simd
		for (index = 0; index < x.size(); index++) {
			x[index] = Ritsu::relu(x[index]);
		}
	}
}

static void BM_MathSigmoid(benchmark::State &state) {
	// Perform setup here
	std::vector<float> x(512 * 512, 1);
	volatile auto sum = 0.0f;
	for (auto _ : state) {
		unsigned int index;
#pragma omp simd
		for (index = 0; index < x.size(); index++) {
			x[index] = Ritsu::computeSigmoid(x[index]);
		}
	}
}

static void BM_TensorSum(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensor({64, 64}, sizeof(float));
	tensor.assignInitValue(1);
	volatile auto sum = 0.0f;
	for (auto _ : state) {
		sum = tensor.sum();
	}
	assert(sum == 64 * 64);
}

static void BM_TensorDot(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensor({64, 64}, sizeof(float));
	tensor.assignInitValue(1);
	volatile auto sum = 0.0f;
	for (auto _ : state) {
		sum = Ritsu::Tensor<float>::innerProduct(tensor, tensor);
	}
	// assert(sum == 64 * 64);
}

static void BM_TensorAddition(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensorA(Ritsu::Shape<uint32_t>({128, 128}));
	Ritsu::Tensor<float> tensorB(Ritsu::Shape<uint32_t>({128, 128}));

	tensorA.assignInitValue(1);
	tensorB.assignInitValue(1);

	Ritsu::Tensor<float> result(Ritsu::Shape<uint32_t>({128, 128}));
	result.assignInitValue(0);

	for (auto _ : state) {
		result += tensorA;
		result += tensorB;
	}

	if (result.sum() == 0) {
		result.flatten();
	}
}

static void BM_TensorMulti(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensorA(Ritsu::Shape<uint32_t>({128, 128}));
	Ritsu::Tensor<float> tensorB(Ritsu::Shape<uint32_t>({128, 128}));
	Ritsu::Tensor<float> result;

	/*  */
	for (auto _ : state) {
		result = Ritsu::Tensor<float>::matrixMultiply(tensorA, tensorB);
	}
	result.flatten();
}

static void BM_TensorAXPY(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensorA(Ritsu::Shape<uint32_t>({128, 128, 1}));
	Ritsu::Tensor<float> tensorB(Ritsu::Shape<uint32_t>({128, 128, 1}));
	const float value = static_cast<float>(rand() % 100);
	Ritsu::Tensor<float> result(Ritsu::Shape<uint32_t>({128, 128, 1}));

	/*  */
	for (auto _ : state) {
		result = (tensorA * value) + tensorB;
	}
	result.flatten();
}

static void BM_LayerRelu(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensorA(Ritsu::Shape<uint32_t>({512, 512, 1}));
	Ritsu::Tensor<float> result(Ritsu::Shape<uint32_t>({512, 512, 1}));
	Ritsu::Relu relu;

	/*  */
	for (auto _ : state) {
		result = relu(tensorA);
	}
	result.flatten();
}

static void BM_LayerSigmoid(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensorA(Ritsu::Shape<uint32_t>({512, 512, 1}));
	Ritsu::Tensor<float> result(Ritsu::Shape<uint32_t>({512, 512, 1}));
	Ritsu::Sigmoid rsigmoidlu;

	/*  */
	for (auto _ : state) {
		result = rsigmoidlu(tensorA);
	}
	result.flatten();
}
static void BM_LayerSoftMax(benchmark::State &state) {
	// Perform setup here
	Ritsu::Tensor<float> tensorA(Ritsu::Shape<uint32_t>({512, 512, 1}));
	Ritsu::Tensor<float> result(Ritsu::Shape<uint32_t>({512, 512, 1}));
	Ritsu::SoftMax softmax;

	/*  */
	for (auto _ : state) {
		result = softmax(tensorA);
	}
	result.flatten();
}

static void BM_ModelAddition(benchmark::State &state) {
	for (auto _ : state) {
		//	result = relu(tensorA);
	}
}

/*  Register the function as a benchmark    */
BENCHMARK(BM_MathProduct);
BENCHMARK(BM_MathSum);
BENCHMARK(BM_MathRelu);
BENCHMARK(BM_MathSigmoid);

/*	*/
BENCHMARK(BM_TensorSum);
BENCHMARK(BM_TensorDot);
BENCHMARK(BM_TensorMulti);
BENCHMARK(BM_TensorAXPY);
BENCHMARK(BM_TensorAddition);

/*	*/
BENCHMARK(BM_LayerRelu);
BENCHMARK(BM_LayerSigmoid);
BENCHMARK(BM_LayerSoftMax);

/*	*/
BENCHMARK(BM_ModelAddition);

/* Run the benchmark    */
BENCHMARK_MAIN();
