#pragma once
#include "Math.h"

namespace Ritsu {

	template <typename T> inline static T computeSigmoid(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return 1.0f / (1.0 + std::exp(-value));
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
	template <typename T> inline static constexpr T computeTanh(const T value) {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		const T e_ = std::exp(-value);
		const T _e_ = std::exp(value);

		return (e_ - _e_) / (e_ + _e_);
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T computeTanhDerivate(T value) {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return std::exp(-value) / std::pow(((std::exp(-value) + 1)), 2);
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

} // namespace Ritsu