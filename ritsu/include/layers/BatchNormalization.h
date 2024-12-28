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
#include "core/Initializers.h"
#include "core/Shape.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class BatchNormalization : public Layer<float> {
	  public:
		BatchNormalization(const std::string &name = "batch normalization") : Layer<float>(name) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override { return tensor; }

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor; }

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override { return tensor; }

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> output({1});
			return output;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->shape = layers[0]->getShape();
			/*	Set input layer */
			this->input = layers[0];
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void build(const Shape<IndexType> &shape) override { /*	Validate */
			// TODO: fix
			Shape<IndexType> weight_shape = shape;

			this->beta = Tensor<DType>(weight_shape);
			ZeroInitializer<DType> zero_init;
			zero_init.set(this->beta);
			this->gamma = Tensor<DType>(weight_shape);
			this->gamma.assignInitValue(1);
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

		std::optional<std::vector<Tensor<DType> *>> getTrainableWeights() noexcept override { return {}; }
		std::optional<std::vector<Tensor<float> *>> getVariables() noexcept override { return {}; }

	  private:
		void compute(const Tensor<float> &input, Tensor<float> &output) {

			const size_t ndims = 10;
			int axis = -1;
			const DType epsilon = std::numeric_limits<DType>::epsilon();
			Tensor<DType> batch_mean = Tensor<DType>::mean(input, axis);
			Tensor<DType> batch_variance = Tensor<DType>::variance(input, batch_mean, axis);

			Tensor<DType> inv = (batch_variance + epsilon).sqrt();
			inv *= gamma;

			/*	*/ // TOOD:fix
			output = input * inv + (this->beta - batch_mean * inv);
		}

	  private:
		Tensor<float> beta;
		Tensor<float> gamma;

		/*	*/
		/*	*/
		Layer<DType> *input{};
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu