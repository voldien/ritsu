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
#include "../Tensor.h"
#include "Object.h"
#include <initializer_list>
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 */
	template <typename T> class Optimizer : public Object {
	  public:
		static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half)");
		using DType = T;
		static constexpr const size_t DTypeSize = sizeof(DType);

	  public:
		Optimizer(const Optimizer &) = delete;
		Optimizer(Optimizer &&) = delete;
		Optimizer &operator=(const Optimizer &) = delete;
		Optimizer &operator=(Optimizer &&) = delete;
		Optimizer(const T learningRate, const std::string &name) noexcept : Object(name) {
			this->setLearningRate(learningRate);
		}
		virtual ~Optimizer() = default;

		/**
		 * @brief
		 */
		virtual void setLearningRate(const T rate) noexcept { this->learningRate = rate; }

		/**
		 * @brief
		 */
		virtual T getLearningRate() const noexcept { return this->learningRate; }

		/**
		 * @brief
		 */
		template <typename... Args> void update_step(const Tensor<T> &gradient, Args &...args) {
			// this->update_step(gradient, {&args...});
		}

		virtual void update_step(const Tensor<T> &gradient, Tensor<T> &variable) = 0;

		/**
		 * @brief
		 */
		virtual void apply_gradients(const Tensor<T> &gradient, Tensor<T> &variable) = 0;

		/**
		 * @brief
		 */
		template <typename... Args> void build(Args &...args) { this->build({&args...}); }

		/**
		 * @brief
		 */
		virtual void build(std::initializer_list<const Tensor<T> &> &list) = 0;

	  private:
		T learningRate;
	};

} // namespace Ritsu