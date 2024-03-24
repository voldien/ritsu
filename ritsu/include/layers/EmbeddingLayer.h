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
#include "../Random.h"
#include "Layer.h"
#include "Tensor.h"
#include <cassert>
#include <cstddef>
#include <ctime>
#include <random>
#include <vector>

namespace Ritsu {
	
	/**
	 * @brief 
	 * 
	 */
	class EmbeddingLayer : public Layer<float> {
		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		EmbeddingLayer(const uint32_t input_dim, const uint32_t output_dim,
					   const std::string &embeddings_initializer = "uniform",
					   const std::string &name = "Embedding Layer")
			: Layer(name) {

			/*	*/
			this->input_dim = input_dim;
			this->output_dim = output_dim;

			/*	*/
			this->shape = {this->input_dim};
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output({this->input_dim}, DTypeSize);

			/*	Verify shape.	*/
			this->compute(tensor, output);

			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {

			/*	Verify shape.	*/

			Tensor<float> inputCopy = tensor;
			this->compute(inputCopy, tensor);

			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> *getTrainableWeights() noexcept override { return &this->weight; }
		Tensor<float> *getVariables() noexcept override { return nullptr; }

		void build(const Shape<IndexType> &shape) override {

			/*	Validate */

			/*	*/
			this->weight = Tensor<float>(
				Shape<IndexType>({static_cast<IndexType>(this->input_dim), static_cast<IndexType>(this->output_dim)}),
				this->DTypeSize);

			/*	*/
			this->initweight();

			/*	*/
			assert(this->weight.getShape().getNrDimensions() == 2);
			assert(this->weight.getShape()[1] == shape[0]);
			assert(this->weight.getShape()[0] == this->input_dim);
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {

			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			this->build(this->getInputs()[0]->getShape());
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			// TODO verify flatten
			Layer<DType> *inputLayer = layers[0];

			if (inputLayer->getShape().getNrDimensions() == 1) {
			}

			this->input = layers[0];

			assert(layers.size() == 1);
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> output(this->weight.getShape());
			computeDerivative(tensor, output);
			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			// computeDerivative(tensor, tensor);
			return tensor;
		}

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:

		// operator
		void compute(const Tensor<float> &inputTesnor, Tensor<float> &output) {
			output = (this->weight % inputTesnor);
		}

		void computeDerivative(const Tensor<float> &error, Tensor<float> &result) {
			result = this->weight * -1.0f; // computeMatrix(this->weight, error);
		}



		void initweight() noexcept {

			RandomNormal<DType> random(0.1, 1.0);
#pragma omp parallel shared(weight)
#pragma omp simd

			for (size_t i = 0; i < this->weight.getNrElements(); i++) {
				this->weight.getValue<DType>(i) = random.rand();
			}
		}

	  private:
		uint32_t input_dim;
		uint32_t output_dim;
		Tensor<float> weight;
	};

} // namespace Ritsu