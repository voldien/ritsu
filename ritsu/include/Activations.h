#pragma once
#include "Math.h"
#include "Tensor.h"

namespace Ritsu {

	template <typename T> inline static T computeSigmoid(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-value));
	}
	template <typename T> inline static T computeSigmoidDerivate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return std::exp(-value) / std::pow(((std::exp(-value) + 1)), 2);
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T relu(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return std::max<T>(0, value);
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T reluDeriviate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return 0;
	}

#pragma omp declare simd uniform(value, alpha)
	template <typename T> inline static constexpr T leakyRelu(T value, const T alpha) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value < 0) {
			return value * alpha;
		}
		return std::max<T>(0, value);
	}

#pragma omp declare simd uniform(value, alpha)
	template <typename T> inline static constexpr T leakyReluDerivative(const T value, const T alpha) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 0;
		}
		return alpha;
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T computeTanh(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		const T e_ = std::exp(-value);
		const T _e_ = std::exp(value);

		return (e_ - _e_) / (e_ + _e_);
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T computeTanhDerivate(T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return std::exp(-value) / std::pow(((std::exp(-value) + static_cast<T>(1))), static_cast<T>(2));
	}

#pragma omp declare simd uniform(coeff, value)
	template <typename T> inline static constexpr T computeLinear(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff * value;
	}

#pragma omp declare simd uniform(coeff)
	template <typename T> inline static constexpr T computeLinearDerivative(const T coeff) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff;
	}

#pragma omp declare simd uniform(coeff, value)
	template <typename T> inline static constexpr T computeExpLinear(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff * value;
	}

#pragma omp declare simd uniform(coeff)
	template <typename T> inline static constexpr T computeExpLinearDerivative(const T coeff) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return coeff;
	}

#pragma omp declare simd uniform(value, beta)
	template <typename T> inline static constexpr T computeSwish(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return value * computeSigmoid<T>(value * beta);
	}

#pragma omp declare simd uniform(value, beta)
	template <typename T> inline static constexpr T computeSwishDerivative(T value, const T beta) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		// TODO:
		return (beta * computeSigmoid(beta, value)) +
			   (sigmoid(value, beta) * (1 - (beta * computeSigmoid(value, beta))));
	}

	template <typename T> void softMax(Tensor &tensor) {
		/*	Iterate through each all elements.    */
		T Inversesum = 0;
		const size_t nrElements = tensor.getNrElements();

#pragma omp parallel
		for (size_t i = 0; i < nrElements; i++) {
			Inversesum += static_cast<T>(std::exp(tensor.getValue<T>(i)));
		}
		Inversesum = 1.0f / Inversesum;
#pragma omp parallel
		for (size_t i = 0; i < nrElements; i++) {
			tensor.getValue<T>(i) = tensor.getValue<T>(i) * Inversesum;
		}
	}

} // namespace Ritsu