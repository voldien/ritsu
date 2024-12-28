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
#include <random>

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

		Random() = default;
		virtual ~Random() = default;

		virtual DType rand() noexcept = 0;

		virtual void reset() noexcept = 0;

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
			this->generator.seed(seed);
		}

		U rand() noexcept override { return this->distribution(this->generator); }
		void reset() noexcept override { this->distribution.reset(); }

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
		RandomNormal(const U mean = 0.0, const U stddev = 1.0, const size_t seed = 118284) {
			this->distribution = std::normal_distribution<U>(mean, stddev);
			std::random_device random_device;
			this->gen = std::mt19937(random_device()); // Standard mersenne_twister_engine seeded with rd()
			this->gen.seed(seed);
		}

		U rand() noexcept override { return this->distribution(this->gen); }
		void reset() noexcept override { this->distribution.reset(); }

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
		RandomBernoulli(const U perc = 1.0, const size_t seed = 512523) {
			this->distribution = std::bernoulli_distribution(perc);
			std::random_device random_device;
			this->generator = std::mt19937(random_device()); // Standard mersenne_twister_engine seeded with rd()
			this->generator.seed(seed);
		}

		U rand() noexcept override { return this->distribution(this->generator); }
		void reset() noexcept override { this->distribution.reset(); }

	  private:
		std::bernoulli_distribution distribution;
		std::mt19937 generator;
	};

	template <typename U> class RandomBinomial : public Random<U> {
	  public:
		RandomBinomial(const U perc = 0.5, const size_t seed = 123456) {
			this->distribution = std::binomial_distribution<>(seed, perc);
			std::random_device random_device;
			this->generator = std::mt19937(random_device());
			this->generator.seed(seed);
		}

		U rand() noexcept override { return this->distribution(this->generator); }
		void reset() noexcept override { this->distribution.reset(); }

	  private:
		std::binomial_distribution<> distribution;
		std::mt19937 generator;
	};

	template <typename U> class RandomPoisson : public Random<U> {
	  public:
		RandomPoisson(const U perc = 0.5, const size_t seed = 123456) {
			this->distribution = std::poisson_distribution<>(seed, perc);
			std::random_device random_device;
			this->generator = std::mt19937(random_device());
			this->generator.seed(seed);
		}

		U rand() noexcept override { return this->distribution(this->generator); }
		void reset() noexcept override { this->distribution.reset(); }

	  private:
		std::poisson_distribution<> distribution;
		std::mt19937 generator;
	};

} // namespace Ritsu