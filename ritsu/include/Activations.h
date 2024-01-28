#pragma once
#include "Tensor.h"
#include <cmath>

namespace Ritsu {

	class ActivactionMath {
	  public:
	};

#pragma omp declare simd uniform(value)
	template <typename T> inline static T computeSigmoid(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return static_cast<T>(1) / (std::exp(-value) + static_cast<T>(1));
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static T computeSigmoidDerivate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		const T sig = computeSigmoid(value);
		return sig * (static_cast<T>(1) - sig);
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
	template <typename T> inline static constexpr T leakyRelu(const T alpha, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value < 0) {
			return alpha * value;
		}
		return std::max<T>(0, value);
	}

#pragma omp declare simd uniform(value, alpha)
	template <typename T> inline static constexpr T leakyReluDerivative(const T alpha, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return alpha;
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T computeTanh(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		// TODO: c++ tanh check
		const T negative_e = static_cast<T>(std::exp(-value));
		const T positive_e = static_cast<T>(std::exp(value));

		return (positive_e - negative_e) / (positive_e + negative_e);
	}

#pragma omp declare simd uniform(value)
	template <typename T> inline static constexpr T computeTanhDerivate(const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return 1.0 - (computeTanh<T>(value) * computeTanh<T>(value));
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
		if (value >= 0) {
			return value;
		}
		return coeff * (std::exp(value) - 1);
	}

#pragma omp declare simd uniform(coeff, value)
	template <typename T> inline static constexpr T computeExpLinearDerivative(const T coeff, const T value) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		if (value >= 0) {
			return 1;
		}
		return coeff * std::exp(value);
	}

#pragma omp declare simd uniform(value, beta)
	template <typename T> inline static constexpr T computeSwish(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		return value * computeSigmoid<T>(value * beta);
	}

#pragma omp declare simd uniform(value, beta)
	template <typename T> inline static constexpr T computeSwishDerivative(const T value, const T beta) noexcept {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		// TODO:
		return (beta * computeSigmoid(beta, value)) +
			   (sigmoid(value, beta) * (1 - (beta * computeSigmoid(value, beta))));
	}

	template <typename T> Tensor<float> &softMax(Tensor<float> &tensor) {
		// TODO: check if subarray exists.
		/*	Iterate through each all elements.    */
		T Inversesum = 0;
		const size_t nrElements = tensor.getNrElements();

#pragma omp parallel for simd reduction(+ : Inversesum)
		for (size_t i = 0; i < nrElements; i++) {
			Inversesum += static_cast<T>(std::exp(tensor.getValue<T>(i)));
		}
		Inversesum = static_cast<T>(1.0) / Inversesum;

#pragma omp parallel for simd
		for (size_t i = 0; i < nrElements; i++) {
			tensor.getValue<T>(i) = tensor.getValue<T>(i) * Inversesum;
		}
		return tensor;
	}

	template <typename T> Tensor<float> &softMaxDerivative(Tensor<float> &tensor) { return tensor; }

} // namespace Ritsu