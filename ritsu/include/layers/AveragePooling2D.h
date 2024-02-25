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
#include <array>
#include <cassert>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class AveragePooling2D : public Layer<float> {

	  public:
		AveragePooling2D(const std::array<uint32_t, 2> &size, const std::string &name = "averagepooling")
			: Layer(name) {
			this->size = size;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override { return tensor; }

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor; }

		void build(const Shape<IndexType> &shape) override {

			this->shape = shape;

			this->shape[-2] /= size[0];
			this->shape[-3] /= size[1];

			/*	*/
			assert(this->getShape().getNrDimensions() == 3);
		}

		void setOutputs(const std::vector<Layer<float> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<float> *> &layers) override { /*	*/

			this->input = layers[0];
		}

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override { return tensorLoss; }
		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override { return tensorLoss; }

	  private:
		std::array<uint32_t, 2> size;

		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu