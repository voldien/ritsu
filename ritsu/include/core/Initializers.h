/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023 Valdemar Lindberg
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */
#pragma once
#include "Random.h"
#include "Tensor.h"
#include "core/Shape.h"

namespace Ritsu {

	/**
	 *
	 */
	template <typename T> class Initializer {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");

	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

		Initializer() {}

		virtual Tensor<DType> get(const Shape<unsigned int> &shape) = 0;
		virtual Tensor<DType> &set(Tensor<DType> &tensor) = 0;

		Tensor<DType> operator()(const Shape<unsigned int> &shape) { return this->get(shape); }
		Tensor<DType> &operator()(Tensor<DType> &tensor) { return this->set(tensor); }
	};

	template <typename T> class RandomNormalInitializer : public Initializer<T> {
	  public:
		RandomNormalInitializer(T mean = 0.0, T stddev = 1.0, int seed = 0) : random(RandomNormal<T>(mean, stddev)) {}

		Tensor<T> get(const Shape<unsigned int> &shape) override {

			Tensor<T> tensor(shape);

			return set(tensor);
		}

		Tensor<T> &set(Tensor<T> &tensor) override {
#pragma omp parallel for simd shared(tensor)
			for (size_t index = 0; index < tensor.getNrElements(); index++) {
				tensor.getValue(index) = this->random.rand();
			}
			return tensor;
		}

	  private:
		RandomNormal<T> random;
	};

	template <typename T> class RandomUniformInitializer : public Initializer<T> {
	  public:
		RandomUniformInitializer(T min = 0.0, T max = 1.0, int seed = 0) : random(RandomUniform<T>(min, max, seed)) {}

		Tensor<T> get(const Shape<unsigned int> &shape) override {

			Tensor<T> tensor(shape);

			return this->set(tensor);
		}

		Tensor<T> &set(Tensor<T> &tensor) override {

#pragma omp parallel for simd shared(tensor)
			for (size_t index = 0; index < tensor.getNrElements(); index++) {
				tensor.getValue(index) = this->random.rand();
			}

			return tensor;
		}

	  private:
		RandomUniform<T> random;
	};

	template <typename T> class ZeroInitializer : public Initializer<T> {
	  public:
		ZeroInitializer() {}

		Tensor<T> get(const Shape<unsigned int> &shape) override {

			Tensor<T> tensor(shape);

			return set(tensor);
		}

		Tensor<T> &set(Tensor<T> &tensor) override {
#pragma omp parallel for simd shared(tensor)
			for (decltype(tensor.getNrElements()) index = 0; index < tensor.getNrElements(); index++) {
				tensor.getValue(index) = 0;
			}

			return tensor;
		}
	};
} // namespace Ritsu