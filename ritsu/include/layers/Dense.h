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
#include "../core/Initializers.h"
#include "Layer.h"
#include "Tensor.h"
#include <cassert>
#include <cstddef>
#include <ctime>
#include <random>
#include <vector>

namespace Ritsu {

	class Dense : public Layer<float> {
		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Dense(uint32_t units, bool use_bias = true,
			  const Initializer<DType> &weight_init = RandomNormalInitializer<DType>(),
			  const Initializer<DType> &bias_init = RandomNormalInitializer<DType>(), const std::string &name = "dense")
			: Layer(name) {

			/*	*/
			this->units = units;
			/*	*/
			this->shape = {this->units};

			/*	*/
			if (use_bias) {
				this->bias = Tensor<float>(Shape<IndexType>({units}));
			}
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output({this->units}, DTypeSize);

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

		Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			/*	*/
			this->build(layer.getShape());

			return *this;
		}

		Tensor<float> *getTrainableWeights() noexcept override { return &this->weight; }
		Tensor<float> *getVariables() noexcept override { return &this->bias; }

		void build(const Shape<IndexType> &shape) override {

			/*	Validate */

			/*	*/
			const Shape<IndexType> weightShape =
				Shape<IndexType>({static_cast<IndexType>(this->units), static_cast<IndexType>(shape[0])});
			this->weight = Tensor<float>(weightShape);

			/*	*/
			this->initweight();
			this->initbias();

			/*	*/
			assert(this->weight.getShape().getNrDimensions() == 2);
			assert(this->weight.getShape()[-1] == shape[0]);
			assert(this->weight.getShape()[-2] == this->units);
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {

			/*	Set input layer */
			this->outputs = layers;

			/*	*/
			Layer<DType> *layer = this->getInputs()[0];
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			// TODO verify flatten
			Layer<DType> *inputLayer = layers[0];
			if (inputLayer->getShape().getNrDimensions() == 1) {
			}

			this->input = inputLayer;

			assert(layers.size() == 1);
		}

		// input

		std::vector<Layer<DType> *> getInputs() const override { return {this->input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return this->outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> output(this->weight.getShape());
			this->computeDerivative(tensor, output);
			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			this->computeDerivative(tensor, tensor);
			return tensor;
		}

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		// operator
		inline void compute(const Tensor<float> &inputTesnor, Tensor<float> &output) const {
			output = (this->weight % inputTesnor) + this->bias;
		}

		void computeDerivative(const Tensor<float> &error, Tensor<float> &result) const {
			// result = computeMatrix(this->weight, error);
			result = this->weight * -1.0f; // TODO: validate
		}

		void initweight() noexcept {
			// TODO improve
			RandomNormal<DType> random(0.1, 1.0);
#pragma omp parallel for simd shared(weight)

			for (size_t i = 0; i < this->weight.getNrElements(); i++) {
				this->weight.getValue<DType>(i) = random.rand();
			}
		}

		void initbias() noexcept {
			// TODO improve
			RandomNormal<DType> random(0.1, 1.0);

#pragma omp parallel shared(bias)
#pragma omp simd
			for (size_t i = 0; i < this->bias.getNrElements(); i++) {
				this->bias.getValue<DType>(i) = random.rand();
			}
		}

	  private:
		Tensor<float> bias;
		uint32_t units;
		Tensor<float> weight;
	};

} // namespace Ritsu