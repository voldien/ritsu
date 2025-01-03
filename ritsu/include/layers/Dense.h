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
#include "../core/Initializers.h"
#include "Layer.h"
#include "RitsuDef.h"
#include "Tensor.h"
#include <cassert>
#include <ctime>
#include <vector>

namespace Ritsu {

	/**
	 */
	class Dense : public Layer<float> {
	  public:
		template <typename U = RandomUniformInitializer<DType>, typename B = ZeroInitializer<DType>>
		Dense(uint32_t units, bool use_bias = true, const U &weight_init2 = RandomUniformInitializer<DType>(-1.0, 1.0),
			  const B &bias_init2 = ZeroInitializer<DType>(), const std::string &name = "dense")
			: Layer(name), units(units), weight_init(new U), bias_init(new B) {

			/*	Assign initilizer.	*/
			*this->weight_init = weight_init2;
			*this->bias_init = bias_init2;

			/*	*/
			if (use_bias) {
				this->bias = Tensor<float>(Shape<IndexType>({units}));
			}
			this->use_bias = use_bias;

			this->variables_reference.reserve(2);
			this->variables_reference = {&this->weight};
			/*	*/
			if (this->use_bias) {
				this->variables_reference.push_back(&this->bias);
			}
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output({this->units}, DTypeSize);
			this->compute(tensor, output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			Tensor<float> inputCopy = tensor;
			this->compute(inputCopy, tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			Tensor<float> output({this->units}, DTypeSize);
			this->compute(tensor, output);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> output({this->units}, DTypeSize);
			this->compute(tensor, output);
			return output;
		}

		std::optional<std::vector<Tensor<DType> *>> getTrainableWeights() noexcept override {
			return this->variables_reference;
		}

		void build(const Shape<IndexType> &buildShape) override {

			if (buildShape.getNrDimensions() > 1) {
				throw InvalidArgumentException("Invalid Shape");
			}

			/*	TODO: Validate */
			const Shape<IndexType> weightShape =
				Shape<IndexType>({static_cast<IndexType>(this->units), static_cast<IndexType>(buildShape[0])});
			this->weight = Tensor<DType>(weightShape);

			/*	*/
			this->shape = {this->units};
			/*	Construct the init values for the weight and bias.	*/
			this->initweight();
			this->initbias();

			/*	Validate.	*/
			assert(this->weight.getShape().getNrDimensions() == 2);
			assert(this->weight.getShape()[-1] == buildShape[0]);
			assert(this->weight.getShape()[-2] == this->units);
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			assert(layers.size() == 1);

			if (layers.size() > 1) {
				/*	*/
				throw InvalidArgumentException("Must only have a single input layer");
			}

			this->input = layers[0];
		}

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

		Tensor<float> compute_gradient(const IndexType parameter_index, const Tensor<float> &deriv_z,
									   const Tensor<float> &Q) override {
			if (parameter_index == 0) {
				return deriv_z.dot(Q.transpose());
			}
			if (parameter_index == 1) {
				return deriv_z.transpose().sum(0);
			}
			return {};
		}

	  private:
		/*	*/
		Layer<DType> *input{};
		std::vector<Layer<DType> *> outputs;

	  protected:
		inline void compute(const Tensor<float> &inputTesnor, Tensor<float> &output) const noexcept {
			/*	*/
			output = (this->weight % inputTesnor);
			if (this->use_bias) {
				output += this->bias;
			}
		}

		inline void computeDerivative(const Tensor<float> &value, Tensor<float> &result) const {
			/*	Dz = W^T*value	*/
			this->weight.transpose().dot(value, result);
		}

		void initweight() noexcept { this->weight_init->set(this->weight); }

		void initbias() noexcept {

			if (this->bias.getNrElements() > 0) {

				this->bias_init->set(this->bias);
				this->bias.reshape({1, this->bias.getShape().getAxisDimensions(-1)});
			}
		}

	  private:
		Tensor<DType> bias;
		uint32_t units;
		Tensor<DType> weight;
		bool use_bias;
		std::vector<Tensor<float> *> variables_reference; /*	*/
		Initializer<DType> *weight_init = nullptr;
		Initializer<DType> *bias_init = nullptr;
	};

} // namespace Ritsu