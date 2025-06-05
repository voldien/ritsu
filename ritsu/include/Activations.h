/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Valdemar Lindberg
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
#include "Tensor.h"
#include <cmath>
#include <limits>

//#define RITSU_FAST_SIGMOID

namespace Ritsu {

#pragma omp declare simd uniform(value) notinbranch
	template <typename T> inline static T computeSigmoid(const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");

#ifdef RITSU_FAST_SIGMOID
		const T sigmoid = value / (1 + std::abs(value));
#else
		const T sigmoid = static_cast<T>(1) / (std::exp(-value) + static_cast<T>(1));
#endif
		return Math::clamp<T>(sigmoid, static_cast<T>(0), static_cast<T>(1));
	}

	template <typename T> inline static void computeSigmoid(T *list, const size_t nrElements) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		size_t index = 0;

#pragma omp simd
		for (index = 0; index < nrElements; index++) {
			list[index] = Ritsu::computeSigmoid<T>(list[index]);
		}
	}

#pragma omp declare simd uniform(value) notinbranch simdlen(8)
	template <typename T> inline static T computeSigmoidDerivative(const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		const T sig = computeSigmoid(value);
		return sig * (static_cast<T>(1) - sig);
	}

	template <typename T> inline static void computeSigmoidDerivative(T *list, const size_t nrElements) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		size_t index = 0;
#pragma omp simd
		for (index = 0; index < nrElements; index++) {
			list[index] = Ritsu::computeSigmoidDerivative<T>(list[index]);
		}
	}

#pragma omp declare simd uniform(value) simdlen(8) notinbranch
	template <typename T> static constexpr T relu(const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		return Math::max<T>(0, value);
	}

	template <typename T> static void relu(T *list, const size_t nrElements) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		size_t index = 0;

#pragma omp simd
		for (index = 0; index < nrElements; index++) {
			list[index] = Ritsu::relu<T>(list[index]);
		}
	}

#pragma omp declare simd uniform(value) simdlen(8)
	template <typename T> static constexpr T reluDerivative(const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return 0;
	}

	template <typename T> static void reluDerivative(T *list, const size_t nrElements) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		size_t index = 0;

#pragma omp simd
		for (index = 0; index < nrElements; index++) {
			list[index] = Ritsu::reluDerivative<T>(list[index]);
		}
	}

#pragma omp declare simd uniform(value, alpha)
	template <typename T> static constexpr T leakyRelu(const T alpha, const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value < 0) {
			return alpha * value;
		}
		return std::max<T>(0, value);
	}

#pragma omp declare simd uniform(value, alpha)
	template <typename T> static constexpr T leakyReluDerivative(const T alpha, const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return alpha;
	}

#pragma omp declare simd uniform(value)
	template <typename T> static constexpr T computeTanh(const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		return std::tanh(value);
	}

#pragma omp declare simd uniform(value)
	template <typename T> static constexpr T computeTanhDerivative(const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		return 1.0 - (computeTanh<T>(value) * computeTanh<T>(value));
	}

#pragma omp declare simd uniform(coeff, value)
	template <typename T> static constexpr T computeLinear(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff * value;
	}

#pragma omp declare simd uniform(coeff)
	template <typename T> static constexpr T computeLinearDerivative(const T coeff) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff;
	}

#pragma omp declare simd uniform(coeff, value)
	template <typename T> static constexpr T computeExpLinear(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return value;
		}
		return coeff * (std::exp(value) - 1);
	}

#pragma omp declare simd uniform(coeff, value)
	template <typename T> static constexpr T computeExpLinearDerivative(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return coeff * std::exp(value);
	}

#pragma omp declare simd uniform(value, beta)
	template <typename T> static constexpr T computeSwish(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		return value * computeSigmoid<T>(beta * value);
	}

#pragma omp declare simd uniform(value, beta)
	template <typename T> static constexpr T computeSwishDerivative(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
					  "Must be a decimal type(float/double/half) or integer.");
		const T fvalue = computeSwish(value, beta);
		const T sigValue = computeSigmoid(value);
		return fvalue + (sigValue * (beta - fvalue));
	}

#pragma omp declare simd
	template <typename T> Tensor<T> &softMax(Tensor<T> &tensor, const int axis = -1) noexcept {
		const size_t nrElements = tensor.getNrElements();

		unsigned int index = 0;

		/*	Compute exponential for each element.	*/
#pragma omp simd simdlen(8)
		for (index = 0; index < nrElements; index++) {
			tensor.template getRawData<T>()[index] = static_cast<T>(std::exp(tensor.template getRawData<T>()[index]));
		}

		/*	Compute inverse sum.	*/
		T Inversesum = 0;
		Inversesum = Math::sum(tensor.getRawData(), nrElements);
		Inversesum = static_cast<T>(1) / Inversesum;

		/*	Apply inverse sum.	*/
#pragma omp simd simdlen(8)
		for (index = 0; index < nrElements; index++) {
			tensor.template getRawData<T>()[index] *= Inversesum;
		}

		/*	*/
		return tensor.clip(static_cast<T>(std::numeric_limits<T>::epsilon()),
						   static_cast<T>(1 - std::numeric_limits<T>::epsilon()));
	}

	template <typename T> Tensor<T> softMax(const Tensor<T> &tensor, const int axis = -1) noexcept {
		Tensor<T> copy = tensor;
		return softMax(const_cast<Tensor<T> &>(tensor), axis);
	}

	template <typename T> Tensor<float> softMaxDerivative(const Tensor<float> &tensor) {
		Tensor<T> diag = Tensor<T>::diag(tensor);

#pragma omp for simd collapse(2)
		for (unsigned int i = 0; i < diag.getShape().getAxisDimensions(0); i++) {
			for (unsigned int j = 0; j < diag.getShape().getAxisDimensions(1); j++) {

				if (i == j) {
					diag.getValue({i, j}) = tensor.getValue(i) * (1 - tensor.getValue(i));
				} else {
					diag.getValue({i, j}) = -tensor.getValue(i) * (1 - tensor.getValue(j));
				}
			}
		}

		return diag;
	}

} // namespace Ritsu