#pragma once
#include <cfloat>
#include <cmath>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Math {
	  public:
		template <class T> inline constexpr static T clamp(T value, T min, T max) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			return Math::max<T>(min, Math::min<T>(max, value));
		}

		/**
		 *	Get max value of a and b.
		 */
		template <typename T> inline constexpr static T max(T value0, T value1) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			return (static_cast<T>(value0) < static_cast<T>(value1)) ? static_cast<T>(value1) : static_cast<T>(value0);
		}

		/**
		 *	Get min value of a and b.
		 */
		template <typename T> inline constexpr static T min(T value0, T value1) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			return (static_cast<T>(value1) < static_cast<T>(value0)) ? static_cast<T>(value1) : static_cast<T>(value0);
		}

		template <typename T> inline constexpr static T frac(T value) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			T part;
			std::modf(value, &part);
			return part;
		}

		template <typename T> constexpr static T sum(const std::vector<T> &list) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Type Must Support addition operation.");
			T sum = 0;
#pragma omp parallel shared(list)
			for (size_t i = 0; i < list.size(); i++) {
				sum += list[i];
			}
			return sum;
		}

		template <typename T> constexpr static T sum(const T *list, size_t nrElements) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Type Must Support addition operation.");
			T sum = 0;
#pragma omp parallel shared(list)
			for (size_t i = 0; i < nrElements; i++) {
				sum += list[i];
			}
			return sum;
		}

		// accuracy.

		template <typename T> constexpr static T mean(const T *list, size_t nrElements) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Type Must Support addition operation.");
			/*	*/
			const T sum = Math::sum<T>(list, nrElements);

			return (static_cast<T>(1) / static_cast<T>(nrElements)) * sum;
		}

		template <typename T> constexpr static T mean(const std::vector<T> &list) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Type Must Support addition operation.");
			const T sum = Math::sum<T>(list);
			return (static_cast<T>(1) / static_cast<T>(list.size())) * sum;
		}

		template <typename T> constexpr static T variance(const T *list, size_t nrElements, T mean) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Type Must Support addition operation.");
			T sum = 0;
			for (size_t i = 0; i < nrElements; i++) {
				sum += (list[i] - mean) * (list[i] - mean);
			}

			return (static_cast<T>(1) / static_cast<T>(nrElements)) * sum;
		}

		template <typename T> constexpr static T variance(const std::vector<T> &list, T mean) noexcept {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Type Must Support addition operation.");
			T sum = 0;

			for (size_t i = 0; i < list.size(); i++) {
				sum += (list[i] - mean) * (list[i] - mean);
			}

			return (static_cast<T>(1) / static_cast<T>(list.size())) * sum;
		}

		/**
		 *	Convert degree to radian.
		 */
		template <typename T> inline constexpr static T degToRad(T deg) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			return deg * static_cast<T>(Deg2Rad);
		}

		/**
		 *	Convert radian to degree.
		 */
		template <typename T> inline constexpr static T radToDeg(T deg) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			return deg * static_cast<T>(Rad2Deg);
		}

		/**
		 *
		 */
		template <typename T> static T wrapAngle(T angle) {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
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
		template <typename T> inline constexpr static T lerp(T value0, T value1, T interp) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			return (value0 + (value1 - value0) * interp);
		}
		template <typename T> inline constexpr static T lerpClamped(T value0, T value1, T interp) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			return (value0 + (value1 - value0) * Math::clamp<T>(interp, static_cast<T>(0.0), static_cast<T>(1.0)));
		}

		template <typename T> inline constexpr static T mod(T value, T mod) noexcept {
			static_assert(std::is_integral<T>::value, "Must be a integer type.");
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

		template <typename T> static inline constexpr T NextPowerOfTwo(T v) {
			static_assert(std::is_integral<T>::value, "Must be a integer type.");
			T res = 1;
			while (res < v) {
				res <<= 1;
			}
			return res;
		}

		/**
		 *
		 */
		template <typename T> static inline constexpr T ClosestPowerOfTwo(T v) {
			T n = NextPowerOfTwo(v);
			T p = 0;
			return 0;
		}

		template <typename T> static inline constexpr bool IsPowerOfTwo(T v) {
			static_assert(std::is_integral<T>::value, "Must be a integer type.");
			return !(v == 0) && !((v - 1) & v);
		}

		/**
		 *	Generate 1D guassian.
		 */
		template <typename T>
		static inline void guassian(std::vector<T> &guassian, unsigned int height, T theta, T standard_deviation) {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			Math::guassian<T>(static_cast<T &>(*guassian.data()), height, theta, standard_deviation);
		}

		template <typename T>
		static void guassian(T &guassian, unsigned int height, T theta, T standard_deviation) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			const T a = (static_cast<T>(1.0) / (standard_deviation * static_cast<T>(std::sqrt(2.0 * Math::PI))));

			/*	*/
			T *pGuass = static_cast<T *>(&guassian);

			for (unsigned int i = 0; i < height; i++) {
				const T b = (-1.0f / 2.0f) * std::pow<T>(((i - standard_deviation) / theta), 2.0f);
				pGuass[i] = a * std::pow<T>(Math::E, b);
			}
		}

		/**
		 *	Generate 2D guassian.
		 */
		template <typename T>
		static inline void guassian(const std::vector<T> &guassian, unsigned int width, unsigned int height, T theta,
									T standard_deviation) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			/*	TODO validate size.	*/

			Math::guassian<T>(static_cast<T &>(*guassian.data()), width, height, theta, standard_deviation);
		}

		template <typename T>
		static inline void guassian(const T &guassian, unsigned int width, unsigned int height, T theta,
									const T standard_deviation) noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			for (unsigned int i = 0; i < height; i++) {
				// guassian(guassian[i * width],)
			}
		}

		template <typename T, typename U> static constexpr inline T gammaCorrection(T x, U gamma) noexcept {

			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			// TODO add support for using vector components.
			T exponent = static_cast<T>(1.0) / gamma;

			return static_cast<T>(std::pow(x, exponent));
		}

		template <typename T> static T gameSpaceToLinear(T gamma, T exponent) noexcept {
			return std::pow(gamma, exponent);
		}

		template <typename T> static inline T random() noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			return {static_cast<T>(::drand48()), static_cast<T>(::drand48())};
		}

		template <typename T> static inline constexpr T align(T size, T alignment) noexcept {
			static_assert(std::is_integral<T>::value, "Must be an integral type.");
			return size + (alignment - (size % alignment));
		}
	};

} // namespace Ritsu