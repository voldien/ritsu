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
#include "Add.h"
#include "Layer.h"
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 */
	class Subtract : public Add {
	  public:
		Subtract(const std::string &name = "subtract") : Add(name) {}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override { return tensor; }

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> output({1});
			return output;
		}

		void build(const Shape<IndexType> &shape) override { this->shape = shape; }

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			if (layers.size() <= 1) {
				throw InvalidArgumentException("Must be greater or equal 2 layers");
			}
			// Check if shape is valid.
			for (size_t i = 0; i < layers.size(); i++) {
			}
			this->inputs = layers;
			this->shape = layers[0]->getShape();
		}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		std::vector<Layer<DType> *> getInputs() const override { return inputs; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		static inline void computeElementSum(Tensor<float> &inputA, const Tensor<float> &inputB) {
			inputA = inputA - inputB;
		}

	  private:
		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu