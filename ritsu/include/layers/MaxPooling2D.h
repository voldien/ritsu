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
#include <algorithm>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class MaxPooling2D : public Layer<float> {

	  public:
		MaxPooling2D(const std::array<uint32_t, 2> &size, const std::string &name = "maxpooling") : Layer(name) {
			this->size = size;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor result;
			this->computeMaxPooling2D(tensor, result);
			return result;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			Tensor result;
			this->computeMaxPooling2D(tensor, result);
			tensor = result;
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		void build(const Shape<IndexType> &buildShape) override {
			this->shape = buildShape;

			this->shape[-2] /= size[0];
			this->shape[-3] /= size[1];

			/*	*/
			assert(this->getShape().getNrDimensions() == 3);
		}

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		void computeMaxPooling2D(const Tensor<float> &tensor, Tensor<float> &output) {

			const size_t width = 0;
			const size_t height = 0;

			// Verify shape

			for (size_t y = 0; y < height; y++) {
				for (size_t x = 0; x < width; x++) {

					DType maxValue = static_cast<DType>(-999999999);
					for (size_t Sy = 0; Sy < this->size[0]; Sy++) {
						for (size_t Sx = 0; Sx < this->size[1]; Sx++) {
							/*	*/
						}
					}
				}
			}
		}

		std::array<uint32_t, 2> size;
	};
} // namespace Ritsu