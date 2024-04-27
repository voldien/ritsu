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
#include "Random.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Dropout : public Layer<float> {

	  public:
		Dropout(const DType perc = 0.5f, const size_t seed = 12345679, const std::string &name = "dropout")
			: Layer(name), perc(perc) {
			this->random = new RandomBernoulli<DType>(perc, seed);
		}
		~Dropout() override { delete this->random; }

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeDropout(tensor);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmpOutput = tensor;
			this->computeDropout(tmpOutput);
			return tmpOutput;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			this->computeDropout(tensor);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> tmpOutput = tensor;
			this->computeDropout(tmpOutput);
			return tmpOutput;
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->shape = layers[0]->getShape();
			this->input = layers[0];
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override {
			Tensor<float> tmpOutput = tensorLoss;
			this->computeDropout(tmpOutput);
			return tmpOutput;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override {
			this->computeDropout(tensorLoss);
			return tensorLoss;
		}

	  private:
		void computeDropout(Tensor<float> &tensor) const { /*	Iterate through each all elements.    */

			this->random->reset();
			const IndexType nrElements = tensor.getNrElements();

#pragma omp parallel for shared(tensor)
			for (IndexType i = 0; i < nrElements; i++) {
				const DType value = tensor.getValue<DType>(i) * this->random->rand() * (1.0f / (1.0 - this->perc));
				tensor.getValue<DType>(i) = value;
			}
		}

		/*	*/
		DType perc;
		Random<DType> *random;
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu