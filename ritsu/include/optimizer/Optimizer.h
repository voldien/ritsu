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
#include "../Tensor.h"
#include "Object.h"
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 */
	template <typename T> class Optimizer : public Object {
	  public:
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		using DType = T;

	  public:
		Optimizer(const T learningRate, const std::string &name) noexcept : Object(name) {
			this->setLearningRate(learningRate);
		}

		/**
		 * @brief Set the Learning Rate object
		 *
		 * @param rate
		 */
		virtual void setLearningRate(const T rate) noexcept { this->learningRate = rate; }

		/**
		 * @brief Get the Learning Rate object
		 *
		 * @return T
		 */
		virtual T getLearningRate() const noexcept { return this->learningRate; }

		/**
		 * @brief
		 *
		 * @param loss
		 * @param variable
		 * @param output_gradient
		 */
		virtual void gradient(const Tensor<float> &loss, const Tensor<float> &variable,
							  Tensor<float> &output_gradient) {}

		/**
		 * @brief
		 *
		 * @param gradient
		 * @param variable
		 */
		virtual void update_step(const Tensor<float> &gradient, Tensor<float> &variable) {}

		/**
		 * @brief
		 *
		 * @param gradient
		 * @param variable
		 */
		virtual void apply(Tensor<float> &gradient, Tensor<float> &variable) {}

	  private:
		T learningRate;
		std::string name;
	};

} // namespace Ritsu