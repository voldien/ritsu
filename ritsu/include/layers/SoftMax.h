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
#include "../Activations.h"
#include "Activation.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class SoftMax : public Activation {
	  public:
		SoftMax(const std::string &name = "softmax") : Activation(name) {}
		~SoftMax() override = default;

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<DType> tmp = tensor;
			Ritsu::softMax<DType>(tmp);
			return tmp;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			Ritsu::softMax<DType>(tensor);
			return tensor;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			Ritsu::softMax<DType>(tensor);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<DType> tmp = tensor;
			Ritsu::softMax<DType>(tmp);
			return tmp;
		}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			return softMaxDerivative<DType>(tensor);
		}
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			tensor = softMaxDerivative<DType>(tensor);
			return tensor;
		}

	  private:
		/*	*/
		Layer<DType> *input = nullptr;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu