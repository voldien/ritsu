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
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 */
	class Tanh : public Activaction {
	  public:
		Tanh(const std::string &name = "tahn") : Activaction(name) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output = tensor;
			this->computeActivation(output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		void build(const Shape<IndexType> &shape) override { this->shape = shape; }

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> output = tensor;
			Tanh::computeDerivative(output);
			return output;
		}
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			Tanh::computeDerivative(tensor);
			return tensor;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  protected:
		static void computeDerivative(Tensor<float> &output) {
			/*Iterate through each all elements.    */
			const size_t nrElements = output.getNrElements();

#pragma omp parallel shared(output)
			for (size_t i = 0; i < nrElements; i++) {
				output.getValue<DType>(i) = Ritsu::computeTanh(output.getValue<DType>(i));
			}
		}

		void computeActivation(Tensor<float> &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = Ritsu::computeTanh(tensor.getValue<DType>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu