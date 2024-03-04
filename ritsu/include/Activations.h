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
#include "Tensor.h"
#include <cmath>

namespace Ritsu {

#pragma omp declare simd uniform(value) simdlen(4)
	template <typename T> inline static T computeSigmoid(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value > 10000) {
			return 1;
		}
		if (value < -10000) {
			return 0;
		}
		return static_cast<T>(1) / (std::exp(-value) + static_cast<T>(1));
	}

#pragma omp declare simd uniform(value) simdlen(4)
	template <typename T> inline static T computeSigmoidDerivate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		const T sig = computeSigmoid(value);
		return sig * (static_cast<T>(1) - sig);
	}

#pragma omp declare simd uniform(value) simdlen(4)
	template <typename T> inline static constexpr T relu(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return Math::max<T>(0, value);
	}

#pragma omp declare simd uniform(value) simdlen(4)
	template <typename T> inline static constexpr T reluDeriviate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return 0;
	}

#pragma omp declare simd uniform(value, alpha) simdlen(4)
	template <typename T> inline static constexpr T leakyRelu(const T alpha, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value < 0) {
			return alpha * value;
		}
		return std::max<T>(0, value);
	}

#pragma omp declare simd uniform(value, alpha) simdlen(4)
	template <typename T> inline static constexpr T leakyReluDerivative(const T alpha, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return alpha;
	}

#pragma omp declare simd uniform(value) simdlen(4)
	template <typename T> inline static constexpr T computeTanh(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		// TODO: c++ tanh check
		const T negative_e = static_cast<T>(std::exp(-value));
		const T positive_e = static_cast<T>(std::exp(value));

		return (positive_e - negative_e) / (positive_e + negative_e);
	}

#pragma omp declare simd uniform(value) simdlen(4)
	template <typename T> inline static constexpr T computeTanhDerivate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return 1.0 - (computeTanh<T>(value) * computeTanh<T>(value));
	}

#pragma omp declare simd uniform(coeff, value) simdlen(4)
	template <typename T> inline static constexpr T computeLinear(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff * value;
	}

#pragma omp declare simd uniform(coeff) simdlen(4)
	template <typename T> inline static constexpr T computeLinearDerivative(const T coeff) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff;
	}

#pragma omp declare simd uniform(coeff, value) simdlen(4)
	template <typename T> inline static constexpr T computeExpLinear(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return value;
		}
		return coeff * (std::exp(value) - 1);
	}

#pragma omp declare simd uniform(coeff, value) simdlen(4)
	template <typename T> inline static constexpr T computeExpLinearDerivative(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return coeff * std::exp(value);
	}

#pragma omp declare simd uniform(value, beta) simdlen(4)
	template <typename T> inline static constexpr T computeSwish(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return value * computeSigmoid<T>(beta * value);
	}

#pragma omp declare simd uniform(value, beta) simdlen(4)
	template <typename T> inline static constexpr T computeSwishDerivative(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		const T fvalue = computeSwish(value, beta);
		const T sigValue = computeSigmoid(value);
		return fvalue + sigValue * (beta - fvalue);
	}

#pragma omp declare simd
	template <typename T> Tensor<T> &softMax(Tensor<T> &tensor, const int axis = -1) noexcept {

		/*	Iterate through each all elements.    */
		T Inversesum = 0;
		const size_t nrElements = tensor.getNrElements();

#pragma omp for simd
		for (size_t i = 0; i < nrElements; i++) {
			Inversesum += static_cast<T>(std::exp(tensor.template getValue<T>(i)));
		}
		Inversesum = static_cast<T>(1) / Inversesum;

#pragma omp for simd
		for (size_t i = 0; i < nrElements; i++) {
			tensor.template getValue<T>(i) = static_cast<T>(std::exp(tensor.template getValue<T>(i))) * Inversesum;
		}
		return tensor;
	}

	template <typename T> Tensor<float> &softMaxDerivative(Tensor<float> &tensor) { return tensor; }

} // namespace Ritsu