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
#include "Layer.h"
#include "Tensor.h"
#include <cassert>
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Regularization : public Layer<float> {
	  public:
		Regularization(const DType L1 = 0, const DType L2 = 0, const std::string &name = "regularization")
			: Layer<float>(name), l1(L1), l2(L2) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> output = std::move(tensor);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			if (this->l1 > 0 && !training) {
				Regularization::computeL1(tensor, this->l1, tensor);
			}
			if (this->l2 > 0 && !training) {
				Regularization::computeL2(tensor, this->l2, tensor);
			}
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> tmp = tensor;
			if (this->l1 > 0 && !training) {
				Regularization::computeL1(tensor, this->l1, tmp);
			}
			if (this->l2 > 0 && !training) {
				Regularization::computeL2(tensor, this->l2, tmp);
			}
			return tmp;
		}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {

			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			// this->build(this->getInputs()[0]->getShape());
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			// TODO verify flatten
			Layer<DType> *inputLayer = layers[0];
			if (inputLayer->getShape().getNrDimensions() == 1) {
			}

			this->input = layers[0];
			this->shape = this->input->getShape();

			assert(layers.size() == 1);
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override {
			Tensor<float> output(tensorLoss.getShape());

			if (this->l1 > 0) {
				Regularization::computeL1(tensorLoss, this->l1, output);
			}
			if (this->l2 > 0) {
				Regularization::computeL2(tensorLoss, this->l2, output);
			}

			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override {

			Tensor<float> output(this->getShape());

			if (this->l1 > 0) {
				computeL1(tensorLoss, this->l1, output);
			}
			if (this->l2 > 0) {
				computeL2(tensorLoss, this->l2, output);
			}

			return tensorLoss;
		}

	  private:
		// TODO: relocate
		static void computeL1(const Tensor<float> &tensor, const DType L1, Tensor<float> &output) noexcept {

			DType sum = 0;
#pragma omp simd reduction(+ : sum)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				sum += std::abs(tensor.getValue<DType>(i));
			}
			sum *= L1;

			output.assign(output);
			output = sum + output;
		}

		static void computeL2(const Tensor<float> &tensor, const DType L2, Tensor<float> &output) noexcept {
			const DType value = L2 * tensor.dot(tensor, -1);
			output.assign(tensor);
			output = value + output;
		}

	  private:
		DType l1;
		DType l2;

		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu