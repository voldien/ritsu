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
#include "RitsuDef.h"
#include <cassert>
#include <functional>
#include <map>

namespace Ritsu {

	/**
	 * @brief
	 */
	template <typename T> class SGD : public Optimizer<T> {
	  public:
		SGD(const T learningRate, const T momentum = 0.0f, const std::string &name = "sgd")
			: Optimizer<T>(learningRate, name) {
			this->momentum = momentum;
		}

		void update_step(const Tensor<T> &gradient, Tensor<T> &variable) override {
			assert(gradient.getShape() == variable.getShape());
			Tensor<T> gradientUpdate = gradient;

			if (gradient.getShape() != variable.getShape()) {
				throw RuntimeException("Invalid Variable and Gradient Shape");
			}

			if (this->momentum > 0) {

				const size_t uid = variable.getUID();
				if (velocities.find(uid) == velocities.end()) {
					velocities[uid] = Tensor<T>::zero(variable.getShape());
				}

				velocities[uid] = (velocities[uid] * this->momentum) + (gradient * (1.0f - this->momentum));

				gradientUpdate = (velocities[uid] * this->getLearningRate());

				this->apply_gradients(gradientUpdate, variable);

			} else {
				gradientUpdate = gradient * this->getLearningRate();

				this->apply_gradients(gradientUpdate, variable);
			}
		}

		void apply_gradients(const Tensor<T> &gradient, Tensor<T> &variable) override {
			assert(gradient.getShape() == variable.getShape());

			if (gradient.getShape() == variable.getShape()) {
				variable -= gradient;
			} else {
				throw RuntimeException("Invalid Variable and Gradient Shape");
			}
		}

		void build(std::initializer_list<const Tensor<void> &> &list) override {}

	  private:
		T momentum;
		std::map<size_t, Tensor<T>> velocities;
	};

} // namespace Ritsu