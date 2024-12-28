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
#include "core/Shape.h"
#include<Openmp/omp-tools.h>

namespace Ritsu {

	/**
	 * @brief
	 */
	class Reshape : public Layer<float> {
	  public:
		Reshape(const Shape<IndexType> &shape, const std::string &name = "reshape")
			: Layer<float>(name), newShape(shape) {}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			tensor.reshape(this->newShape);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmp = tensor;
			tmp.reshape(this->newShape);
			return tmp;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			tensor.reshape(this->newShape);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> tmp = tensor;
			tmp.reshape(this->newShape);
			return tmp;
		}

		void build(const Shape<IndexType> &shape) override {
			//			assert(shape == this->newShape);
			this->shape = newShape;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			Layer<DType> *layer = layers[0];
			/*	*/
			if (layer->getShape().getNrElements() != this->getShape().getNrElements()) {
				/*	*/
			}

			this->input = layers[0];
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<DType> compute_derivative(const Tensor<DType> &tensor) override {
			Tensor<DType> copy = tensor;
			const Tensor<DType>::IndexType batchSize = tensor.getShape()[0];
			return copy.reshape(this->input->getShape());
		}

		Tensor<DType> &compute_derivative(Tensor<DType> &tensor) const override {
			const Tensor<DType>::IndexType batchSize = tensor.getShape()[0];
			return tensor.reshape(this->input->getShape());
		}

	  private:
		Layer<DType> *input = nullptr;
		std::vector<Layer<DType> *> outputs;
		Shape<IndexType> newShape;
	};
} // namespace Ritsu