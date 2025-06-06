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
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Math {
	  public:
		/**
		 *
		 */
#pragma omp declare simd uniform(value) notinbranch
		template <typename T> constexpr static T abs(const T value) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Must be a decimal type(float/double/half) or integer.");

			if constexpr (std::is_integral_v<T>) {
				return std::abs(static_cast<long>(value));
			} else if constexpr (std::is_floating_point_v<T>) {
				return std::fabs(value);
			} else {
				assert(0);
			}
		}

		/**
		 *
		 */
#pragma omp declare simd uniform(value, min, max) notinbranch
		template <typename T> static constexpr T clamp(const T value, const T min, const T max) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Must be a decimal type(float/double/half) or integer.");
			return Math::max<T>(min, Math::min<T>(max, value));
		}

		/**
		 *	Get max value of a and b.
		 */
#pragma omp declare simd uniform(value0, value1) notinbranch
		template <typename T> static constexpr T max(const T value0, const T value1) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Must be a decimal type(float/double/half) or integer.");
			return (static_cast<T>(value0) < static_cast<T>(value1)) ? static_cast<T>(value1) : static_cast<T>(value0);
		}

		/**
		 *	Get min value of a and b.
		 */
#pragma omp declare simd uniform(value0, value1) notinbranch
		template <typename T> static constexpr T min(const T value0, const T value1) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Must be a decimal type(float/double/half) or integer.");
			return (static_cast<T>(value1) < static_cast<T>(value0)) ? static_cast<T>(value1) : static_cast<T>(value0);
		}
		

		template <typename T> constexpr static T frac(const T value) noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			T part;
			std::modf(value, &part);
			return part;
		}

		template <typename T> static T sum(const std::vector<T> &list) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Type Must Support addition operation.");
			return Math::sum<T>(list.data(), list.size());
		}

#pragma omp declare simd
		template <typename T> static T sum(const T *list, const size_t nrElements) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Type Must Support addition operation.");
			T sum = 0;
			size_t index = 0;
			
#pragma omp simd reduction(+ : sum)
			for (index = 0; index < nrElements; index++) {
				sum += list[index];
			}

			return sum;
		}

		template <typename T> static T sum_abs(const std::vector<T> &list) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Type Must Support addition operation.");
			return Math::sum_abs<T>(list.data(), list.size());
		}

		template <typename T> static T sum_abs(const T *list, const size_t nrElements) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T> || std::is_enum_v<T>,
						  "Type Must Support addition operation.");
			T abs_sum = 0;
			T value;
			size_t index = 0;
#pragma omp simd reduction(+ : abs_sum) private(value)
			for (index = 0; index < nrElements; index++) {
				value = list[index];
				abs_sum += std::abs(value);
			}
			return abs_sum;
		}

		template <typename T> static T product(const std::vector<T> &list) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			return Math::product<T>(list.data(), list.size());
		}

#pragma omp declare simd
		template <typename T> static T product(const T *list, const size_t nrElements) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			T product_combined = 1;
			size_t index = 0;

#pragma omp simd reduction(* : product_combined)
			for (index = 0; index < nrElements; index++) {
				product_combined *= list[index];
			}
			return product_combined;
		}

#pragma omp declare simd uniform(nrElements)
		template <typename T> static T dot(const T *listA, const T *listB, const size_t nrElements) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			T sum = 0;
			size_t index = 0;

#pragma omp simd reduction(+ : sum)
			for (index = 0; index < nrElements; index++) {
				sum += listA[index] * listB[index];
			}
			return sum;
		}

#pragma omp declare simd uniform(exponent, nrElements)
		template <typename T> static inline void pow(const T exponent, T *list, const size_t nrElements) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			/*	*/
			size_t index = 0;

#pragma omp for simd
			for (index = 0; index < nrElements; index++) {
				list[index] = static_cast<T>(std::pow(list[index], exponent));
			}
		}

#pragma omp declare simd
		template <typename T> constexpr static T mean(const T *list, const size_t nrElements) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			/*	*/
			const T sum = Math::sum<T>(list, nrElements);
			const float averageInverse = static_cast<float>(1) / static_cast<float>(nrElements);
			return static_cast<T>(averageInverse * sum);
		}

		template <typename T> constexpr static T mean(const std::vector<T> &list) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			const T sum = Math::sum<T>(list);
			const float averageInverse = static_cast<float>(1) / static_cast<float>(list.size());
			return static_cast<T>(averageInverse * sum);
		}

#pragma omp declare simd uniform(mean)
		template <typename T> static T variance(const T *list, const size_t nrElements, const T mean) noexcept {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			T sum = 0;
			size_t index = 0;
#pragma omp simd reduction(+ : sum)
			for (index = 0; index < nrElements; index++) {
				sum += (list[index] - mean) * (list[index] - mean);
			}

			return (static_cast<T>(1) / static_cast<T>((nrElements - 1))) * sum;
		}

		template <typename T> static T variance(const std::vector<T> &list, const T mean) {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			return Math::variance<T>(list.data(), list.size(), mean);
		}

		template <typename T> constexpr static T standardDeviation(const std::vector<T> &list, const T mean) {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");
			return static_cast<T>(std::sqrt(Math::variance<T>(list, mean)));
		}

#pragma omp declare simd uniform(meanA, meanB)
		template <typename T>
		static T cov(const std::vector<T> &listA, const std::vector<T> &listB, const T meanA, const T meanB) {
			static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
						  "Type Must Support addition operation.");

			T sum = 0;

			const size_t nrElements = listB.size();
#pragma omp simd reduction(+ : sum)
			for (size_t i = 0; i < nrElements; i++) {
				sum += (listA[i] - meanA) * (listB[i] - meanB);
			}

			return (static_cast<T>(1) / static_cast<T>(listA.size())) * sum;
		}

		template <typename T>
		constexpr static T cor(const std::vector<T> &listA, const std::vector<T> &listB, const T meanA, const T meanB) {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");

			return cov<T>(listA, listB, meanA, meanB) /
				   static_cast<T>(std::sqrt(variance<T>(listB, meanB) * variance<T>(listA, meanA)));
		}

		/**
		 *	Convert degree to radian.
		 */
		template <typename T> constexpr static T degToRad(const T deg) noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			return deg * static_cast<T>(Deg2Rad);
		}

		/**
		 *	Convert radian to degree.
		 */
		template <typename T> constexpr static T radToDeg(const T rad) noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			return rad * static_cast<T>(Rad2Deg);
		}

		/**
		 *
		 */
		template <typename T> static T wrapAngle(T angle) noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			while (angle > static_cast<T>(Math::PI_2)) {
				angle -= static_cast<T>(Math::PI_2);
			}
			while (angle < 0.0f) {
				angle += static_cast<T>(Math::PI_2);
			}
			return angle;
		}

		/**
		 * @brief Linear interpolation.
		 *
		 * @tparam T
		 * @param a Start point.
		 * @param b End point.
		 * @param t normalized interpolation, between [0,1], a value greater than 1 will not be clamped
		 * 	and will thus exceed eitehr the start or the end point.
		 * @return constexpr T
		 */
#pragma omp declare simd uniform(value0, value1, interp)
		template <typename T, typename U>
		constexpr static T lerp(const T value0, const T value1, const U interp) noexcept {
			static_assert(std::is_floating_point_v<U>, "Must be a decimal type(float/double/half).");
			return (value0 + ((value1 - value0) * interp));
		}

#pragma omp declare simd uniform(value0, value1, interp)
		template <typename T, typename U>
		constexpr static T lerpClamped(const T value0, const T value1, const U interp) noexcept {
			static_assert(std::is_floating_point_v<U>, "Must be a decimal type(float/double/half).");
			return (value0 + ((value1 - value0) * Math::clamp<U>(interp, static_cast<U>(0.0), static_cast<U>(1.0))));
		}

#pragma omp declare simd uniform(value, mod)
		template <typename T> constexpr static T mod(const T value, const T mod) noexcept {
			static_assert(std::is_integral_v<T>, "Must be a integer type.");
			return (value % mod + mod) % mod;
		}

		/**
		 *
		 */
		static constexpr double E = 2.718281828459045235;
		static constexpr double PI = 3.141592653589793238462643383279502884;
		static constexpr double PI_half = Math::PI / 2.0;
		static constexpr double PI_2 = Math::PI * 2.0;
		static constexpr double Epsilon = FLT_EPSILON;
		static constexpr double Deg2Rad = Math::PI / 180.0;
		static constexpr double Rad2Deg = 180.0 / Math::PI;
		static constexpr double NegativeInfinity = 0;

		template <typename T> static constexpr T NextPowerOfTwo(const T value) noexcept {
			static_assert(std::is_integral_v<T>, "Must be a integer type.");
			T res = 1;
			while (res < value) {
				res <<= 1;
			}
			return res;
		}

		/**
		 *
		 */
		template <typename T> static constexpr T ClosestPowerOfTwo(const T value) noexcept {
			T n = NextPowerOfTwo(value);
			T p = 0;
			return 0;
		}

		template <typename T> static constexpr bool IsPowerOfTwo(const T value) noexcept {
			static_assert(std::is_integral_v<T>, "Must be a integer type.");
			return !(value == 0) && !((value - 1) & value);
		}

		/**
		 *	Generate 1D guassian.
		 */
		template <typename T>
		static void guassian(std::vector<T> &guassian, const unsigned int height, const T theta,
							 const T standard_deviation) {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			guassian.reserve(height);
			Math::guassian<T>(guassian.data(), height, theta, standard_deviation);
		}

		template <typename T>
		static void guassian(T *guassian, const unsigned int height, const T theta, const T standard_deviation) {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");

			const T exp_inverse =
				(static_cast<T>(1.0) / (static_cast<T>(2.0) * standard_deviation * standard_deviation));
			const T sqr_2_pi_inverse = 1.0 / (standard_deviation * static_cast<T>(std::sqrt(2 * Math::PI)));

			const T offset = static_cast<T>(height) / -2;
			// #pragma omp simd
			for (unsigned int i = 0; i < height; i++) {

				const T exp_num_sqrt = (i - theta + offset);

				const T exponent = exp_inverse * -(exp_num_sqrt * exp_num_sqrt);
				const T value = sqr_2_pi_inverse * std::exp(exponent);

				guassian[i] = value;
			}
		}

		/**
		 *	Generate 2D guassian.
		 */
		template <typename T>
		static void guassian(std::vector<T> &guassian, const unsigned int width, const unsigned int height,
							 const T theta, const T standard_deviation) {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			guassian.reserve(width * height);
			Math::guassian<T>(static_cast<T &>(*guassian.data()), width, height, theta, standard_deviation);
		}

		template <typename T>
		static void guassian(const T &guassian, const unsigned int width, const unsigned int height,
							 const T standard_deviation) noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");

			const T exp_inverse =
				(static_cast<T>(1.0) / (static_cast<T>(2.0) * standard_deviation * standard_deviation));
			const T sqr_2_pi_inverse = 1.0 / (standard_deviation * static_cast<T>(std::sqrt(2 * Math::PI)));

			const T offset = static_cast<T>(height) / -2;

			for (unsigned int y = 0; y < height; y++) {
				for (unsigned int x = 0; x < width; x++) {
					const T exponent = exp_inverse * -((x * x) + (y * y));

					guassian[(y * width) + x] = sqr_2_pi_inverse * std::exp(exponent);
				}
			}
		}

		template <typename T> std::vector<T> PCA(std::vector<T> &data) {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");

			Math::PCA<T>(data.data(), data.size());
		}

		template <typename T> std::vector<T> PCA(const T *list, const size_t nrElements) {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");

			const T nInverse = (1.0 / static_cast<T>(nrElements));

			const T m = Math::mean<T>(list, nrElements);

			// Math::cov(list, const std::vector<T> &listB, const T meanA, const T meanB)
			// cov
			// Matrix3x3 C = nInverse;
		}

		template <typename T, typename U> static constexpr T gamma(const T value, const U gamma) noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");

			const T exponent = static_cast<T>(1) / gamma;
			return static_cast<T>(std::pow(value, exponent));
		}

		template <typename T> static T gameSpaceToLinear(const T gamma, const T exponent) noexcept {
			return std::pow(gamma, exponent);
		}

		template <typename T> static T random() noexcept {
			static_assert(std::is_floating_point_v<T>, "Must be a decimal type(float/double/half).");
			return {static_cast<T>(::drand48()), static_cast<T>(::drand48())};
		}

#pragma omp declare simd uniform(size, alignment) notinbranch
		template <typename T> static constexpr T align(const T size, const T alignment) noexcept {
			static_assert(std::is_integral_v<T>, "Must be an integral type.");
			return size + (alignment - (size % alignment));
		}
	};

} // namespace Ritsu