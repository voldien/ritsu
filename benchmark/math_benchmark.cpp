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
	Ritsu::Tensor<float> result;

	/*  */
	for (auto _ : state) {
		result = (tensorA * value) + tensorB;
	}
	result.flatten();
}

/*  Register the function as a benchmark    */
BENCHMARK(BM_MathProduct);
BENCHMARK(BM_MathSum);
BENCHMARK(BM_TensorSum);
BENCHMARK(BM_TensorDot);

BENCHMARK(BM_TensorMulti);
BENCHMARK(BM_TensorAXPY);
BENCHMARK(BM_TensorAddition);

/* Run the benchmark    */
BENCHMARK_MAIN();
