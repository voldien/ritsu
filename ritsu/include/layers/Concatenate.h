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

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Concatenate : public Layer<float> {

	  public:
		Concatenate(Layer<DType> &a, Layer<DType> &b, const std::string &name = "concatenate")
			: Concatenate({&a, &b}, name) {}
		Concatenate(const std::vector<Layer<DType> *> &layers, const std::string &name) : Layer<float>(name) {
			this->inputs = layers;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override { return tensor; }

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> output({1});
			return output;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->inputs = layers;

			Shape<IndexType> newShape = inputs[0]->getShape();
			for (size_t i = 1; i < inputs.size(); i++) {
				newShape.append(inputs[i]->getShape());
			}

			this->shape = newShape;
		}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override { return tensorLoss; }
		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override { return tensorLoss; }

	  private:
		static void concatenate(const Tensor<float> &tensorA, const Tensor<float> &tensorB, Tensor<float> &output) {
			Tensor<float> copyA = tensorA;
			copyA.concatenate(tensorB);
			output = copyA;
		}

		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu