#pragma once
#include "Random.h"
#include "Tensor.h"
#include "core/Shape.h"
#include <random>
#include <string>
#include <vector>

namespace Ritsu {

	template <typename T> class Initializer {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");

	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

		Initializer() {}

		virtual Tensor<float> get(const Shape<unsigned int> &shape) = 0;

		Tensor<float> &operator()(const Shape<unsigned int> &shape) { return this->get(shape); }
	};

	template <typename T> class RandomNormalInitializer : public Initializer<T> {
	  public:
		RandomNormalInitializer(T mean = 0.0, T stddev = 1.0, int seed = 0) : random(RandomNormal<T>(mean, stddev)) {}

		Tensor<float> get(const Shape<unsigned int> &shape) override {

			Tensor<float> tensor(shape, sizeof(T));

#pragma omp parallel for simd shared(tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<T>(i) = this->random.rand();
			}

			return tensor;
		}

	  private:
		RandomNormal<T> random;
	};

	template <typename T> class ZeroInitializer : public Initializer<T> {
	  public:
		ZeroInitializer() {}

		Tensor<float> get(const Shape<unsigned int> &shape) override {

			Tensor<float> tensor(shape, sizeof(T));
#pragma omp parallel for simd shared(tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<T>(i) = 0;
			}

			return tensor;
		}
	};
} // namespace Ritsu