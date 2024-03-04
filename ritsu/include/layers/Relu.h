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
#include "Activaction.h"
#include "Tensor.h"
#include "layers/Layer.h"
#include <cassert>
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Relu : public Activaction {
	  public:
		Relu(const std::string &name = "relu") : Activaction(name) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> output = tensor; // Copy
			this->computeReluActivation(output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeReluActivation(tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeReluActivation(tensor);
			return tensor;
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

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override {
			Tensor<float> output;
			computeDeriviate(output);
			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override {
			computeDeriviate(tensorLoss);
			return tensorLoss;
		}

	  protected:
		void computeReluActivation(Tensor<float> &tensor) {

			const IndexType nrElements = tensor.getNrElements();
#pragma omp parallel for shared(tensor)
			for (IndexType i = 0; i < nrElements; i++) {
				const DType value = relu<DType>(tensor.getValue<DType>(i));
				tensor.getValue<DType>(i) = value;
			}
		}

		static void computeDeriviate(Tensor<float> &tensor) {
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel for shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = reluDeriviate(tensor.getValue<DType>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu