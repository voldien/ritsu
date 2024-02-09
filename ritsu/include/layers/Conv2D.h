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
#include "../core/Initializers.h"
#include "Layer.h"
#include <cstdint>
#include <ctime>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Conv2D : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Conv2D(const uint32_t filters, const std::vector<uint32_t> &kernel_size, const std::array<uint32_t, 2> &stride,
			   const std::string &padding, bool useBias = true,
			   const Initializer<DType> &kernel_init = RandomNormalInitializer<DType>(),
			   const Initializer<DType> &bias_init = RandomNormalInitializer<DType>(),
			   const std::string &name = "Conv2D")
			: Layer<float>(name) {

			this->filters = filters;
			this->stride = stride;
			this->kernel = kernel_size;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output(getShape());
			this->computeConv2D(tensor, output);

			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeConv2D(tensor, tensor);
			return tensor;
		}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			this->computeConv2D(tensor, tensor);
			return tensor;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void build(const Shape<IndexType> &shape) override {
			this->initbias(shape);
			this->initKernels(shape);
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  protected:
		// operator

		void computeConv2D(const Tensor<float> &input, Tensor<float> &output) { /*	*/

			const size_t nrFilters = this->getNrFilters();

#pragma omp parallel for simd shared(_kernelWeight)
			for (size_t i = 0; i < nrFilters; i++) {
				// TODO add matrix multiplication.

				for (size_t x = 0; x < 1; x++) {
					for (size_t y = 0; y < 1; y++) {
						_kernelWeight.getValue<float>(nrFilters * this->kernel.getNrElements());
					}
				}
			}
		}

		void initKernels(const Shape<IndexType> &shape) noexcept {}

		void initbias(const Shape<IndexType> &shape) noexcept {
			// std::srand(std::time(NULL));
			// for (int i = 0; i < bias.size(); i++) {
			//	bias[i] = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			//}
		}

	  protected:
		size_t getNrFilters() const noexcept { return this->filters; }

	  private:
		size_t filters;
		std::vector<DType> bias;
		Tensor<float> _bias;
		Tensor<float> _kernelWeight;
		Shape<IndexType> kernel;
		std::array<uint32_t, 2> stride;
		std::vector<DType> weight;
	};
} // namespace Ritsu