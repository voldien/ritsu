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
#include "Optimizer.h"
#include <cassert>
#include <functional>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class SGD : public Optimizer<T> {
	  public:
		SGD(const T learningRate, const T momentum = 0.0f, const std::string &name = "sgd")
			: Optimizer<T>(learningRate, name) {
			this->momentum = momentum;
		}

		void gradient(const Tensor<float> &loss, const Tensor<float> &variable,
					  Tensor<float> &output_gradient) override {}

		void update_step(const Tensor<float> &gradient, Tensor<float> &variable) override {

			Tensor<float> tmpGradient = gradient;

			if (momentum > 0) {
				// velocity = momentum * velocity - (gradient * this->getLearningRate());
				// variable = variable + velocity;
			} else {
				Tensor<float> gradientUpdate = tmpGradient * this->getLearningRate();

				/*	*/
				std::cout << gradientUpdate.getShape() << " " << variable.getShape() << std::flush;

				// TODO: check and validate.
				assert(gradientUpdate.getShape() == variable.getShape());
				/*	Verify the shape.	*/
				if (gradientUpdate.getShape() == variable.getShape()) {

					variable = gradientUpdate;
				} else {
				}
			}
		}

	  private:
		T momentum;
		T velocity;
	};

} // namespace Ritsu