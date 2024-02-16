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
#include "../Math.h"
#include "Layer.h"
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

		Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->shape = layers[0]->getShape();
			/*	Set input layer */
			this->input = layers[0];
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			this->build(this->getInputs()[0]->getShape());
		}
		void build(const Shape<IndexType> &shape) override { /*	Validate */
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

		Tensor<float> *getTrainableWeights() noexcept override { return nullptr; }
		Tensor<float> *getVariables() noexcept override { return nullptr; }

	  private:
		void compute(const Tensor<float> &input, Tensor<float> &output) {

			const size_t ndims = 10;

			for (size_t i = 0; i < ndims; i++) {

				Tensor<float> subset = input.getSubset(0, 12, Shape<IndexType>({12}));
				DType mean = Math::mean(subset.getRawData<DType>(), subset.getNrElements());
				// TODO add // (subset - mean) /
				(Math::variance<DType>(subset.getRawData<DType>(), subset.getNrElements(), mean) + 0.00001f);
			}

			/*	*/
		}

	  private:
		Tensor<float> beta;
		Tensor<float> alpha;

		/*	*/
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu