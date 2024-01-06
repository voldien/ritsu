#pragma once
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

		virtual Tensor get(const Shape<unsigned int> &shape) = 0;

		Tensor &operator()(const Shape<unsigned int> &shape) { return this->get(shape); }
	};

	template <typename T> class RandomNormalInitializer : public Initializer<T> {
		RandomNormalInitializer(T mean = 0.0, T stddev = 0.05, int seed = 0) {}

		Tensor get(const Shape<unsigned int> &shape) override {

			Tensor tensor(shape);
#pragma omp parallel shared(tensor)
#pragma omp simd
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<T>(i) = 0;
			}

            return tensor;
		}
	}; // namespace Ritsu
} // namespace Ritsu