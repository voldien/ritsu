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

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Flatten : public Layer<float> {
	  public:
		Flatten(const std::string &name = "flatten") : Layer<float>(name) {}

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor.flatten(); }

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmp = tensor;
			return tmp.flatten();
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override { return tensor.flatten(); }

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> tmp = tensor;
			return tmp.flatten();
		}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape.flatten(); }

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];

			this->shape = this->input->getShape().flatten();
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<DType> copy = tensor;
			const Tensor<DType>::IndexType batchSize = tensor.getShape()[0];
			return copy.assignInitValue(1); //.reshape(this->input->getShape());
		}
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			return tensor.assignInitValue(1); //.reshape(this->input->getShape());
		}

	  private:
		Layer<DType> *input{};
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu