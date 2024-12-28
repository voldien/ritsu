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
	 * @brief Scale value.
	 *
	 */
	class Rescaling : public Layer<float> {
	  public:
		Rescaling(const DType scale, const std::string &name = "rescaling") : Layer<float>(name), scale(scale) {}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeScale(tensor);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmp = tensor;
			this->computeScale(tmp);
			return tmp;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			this->computeScale(tensor);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> tmp = tensor;
			this->computeScale(tmp);
			return tmp;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];
			this->shape = this->input->getShape();
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		/*	*/
		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		/*	*/
		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor * (1.0f / this->scale); }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			return tensor * (1.0f / this->scale);
		}

	  protected:
		inline void computeScale(Tensor<float> &tensor) const noexcept { tensor = tensor * this->scale; }

	  private:
		DType scale;
		Layer<DType> *input{};
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu