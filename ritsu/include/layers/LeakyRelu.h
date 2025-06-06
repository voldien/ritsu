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
#include "Layer.h"
#include <ctime>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class LeakyRelu : public Layer<float> {
	  public:
		LeakyRelu(const DType alpha, const std::string &name = "leaky-relu") : Layer<DType>(name), alpha(alpha) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> output = tensor; // Copy
			this->computeReluLeakyActivation(output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeReluLeakyActivation(tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeReluLeakyActivation(tensor);
			return tensor;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			this->computeReluLeakyActivation(tensor);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> output = tensor; // Copy
			this->computeReluLeakyActivation(output);
			return output;
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->shape = layers[0]->getShape();
			this->input = layers[0];
		}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> output = tensor;
			this->computeReluLeakyDerivative(output);
			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			this->computeReluLeakyDerivative(tensor);
			return tensor;
		}

	  protected:
		void computeReluLeakyActivation(Tensor<float> &tensor) const {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel for shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = leakyRelu(tensor.getValue<DType>(i), this->alpha);
			}
		}

		void computeReluLeakyDerivative(Tensor<float> &tensor) const {
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel for shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = leakyReluDerivative(tensor.getValue<DType>(i), this->alpha);
			}
		}

	  private:
		DType alpha;
		Layer<DType> *input{};
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu