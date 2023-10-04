#pragma once
#include <random>
#include <string>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class Random {
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");

	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

		Random() {}
		virtual ~Random() = default;

		virtual DType rand() = 0;

	  protected:
		std::random_device random_device; // Will be used to obtain a seed for the random number engine
	};

	/**
	 * @brief 
	 * 
	 * @tparam U 
	 */
	template <typename U> class RandomUniform : public Random<U> {
	  public:
		RandomUniform(const U min = 0.0, const U max = 1.0, const size_t seed = 0) {
			this->distribution = std::uniform_real_distribution<U>(min, max);
		}

		U rand() override { return this->distribution(this->generator); }

	  private:
		std::uniform_real_distribution<U> distribution;
		std::default_random_engine generator;
	};

	/**
	 * @brief 
	 * 
	 * @tparam U 
	 */
	template <typename U> class RandomNormal : public Random<U> {
	  public:
		RandomNormal(const U mean = 0.0, const U stddev = 1.0, const size_t seed = 0) {
			this->distribution = std::normal_distribution<U>(mean, stddev);
			std::random_device random_device;
			this->gen = std::mt19937(random_device()); // Standard mersenne_twister_engine seeded with rd()
		}

		U rand() override { return this->distribution(this->gen); }

	  private:
		std::normal_distribution<U> distribution;
		std::mt19937 gen;
	};

	/**
	 * @brief 
	 * 
	 * @tparam U 
	 */
	template <typename U> class RandomBernoulli : public Random<U> {
	  public:
		RandomBernoulli(const U perc = 1.0, const size_t seed = 0) {
			this->distribution = std::bernoulli_distribution(perc);
			std::random_device random_device;
			this->generator =
				std::default_random_engine(random_device()); // Standard mersenne_twister_engine seeded with rd()
		}

		U rand() override { return this->distribution(this->generator); }

	  private:
		std::bernoulli_distribution distribution;
		std::default_random_engine generator;
	};

} // namespace Ritsu