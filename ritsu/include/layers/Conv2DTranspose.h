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
#include "Conv2D.h"
#include <cstdint>
#include <ctime>
#include <vector>

namespace Ritsu {

	class Conv2DTranspose : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Conv2DTranspose(const size_t filters, const std::array<uint32_t, 2> &kernel_size,
						const std::array<uint32_t, 2> &stride = {1, 1}, const ConvPadding padding = ConvPadding::Valid,
						const std::string name = "conv2Dtranspose")
			: Layer<float>(name) {}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override { return tensorLoss; }
		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override { return tensorLoss; }

	  protected:
		size_t getNrFilters() const noexcept { return this->filters; }

	  private:
		size_t filters;
	};
} // namespace Ritsu