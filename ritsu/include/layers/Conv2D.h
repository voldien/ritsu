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

	enum class ConvPadding { Same, Valid };
	/**
	 * @brief
	 *
	 */
	class Conv2D : public Layer<float> {

	  public:
		Conv2D(const uint32_t filters, const std::array<uint32_t, 2> &kernel_size,
			   const std::array<uint32_t, 2> &stride, const ConvPadding padding = ConvPadding::Valid,
			   bool useBias = true, const Initializer<DType> &kernel_init = RandomNormalInitializer<DType>(),
			   const Initializer<DType> &bias_init = ZeroInitializer<DType>(), const std::string &name = "Conv2D")
			: Layer<float>(name) {

			this->filter_count = filters;
			this->stride_size = stride;
			this->kernel_size = kernel_size;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output;
			this->computeConv2D(tensor, output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			Tensor<float> inputCopy = tensor;
			this->computeConv2D(inputCopy, tensor);
			return tensor;
		}

		void build(const Shape<IndexType> &buildShape) override {

			if (buildShape.getNrDimensions() > 3) {
				throw InvalidArgumentException("Invalid Shape");
			}

			this->shape = buildShape;

			/*	Down scale image.	*/
			this->shape[-1] = filter_count;
			this->shape[-2] = (IndexType)(static_cast<float>(this->shape[-2]) / static_cast<float>(stride_size[0]));
			this->shape[-3] = (IndexType)(static_cast<float>(this->shape[-3]) / static_cast<float>(stride_size[1]));

			/*	*/
			assert(this->getShape().getNrDimensions() == 3);

			this->initbias(shape);
			this->initKernels(shape);
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		std::vector<Layer<DType> *> getInputs() const override { return {this->input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return this->outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

		Tensor<float> *getTrainableWeights() noexcept override { return &this->filters; }
		Tensor<float> *getVariables() noexcept override { return &this->bias; }

	  protected:
		// operator

		void computeConv2D(const Tensor<float> &input, Tensor<float> &output) { /*	*/

			// Assert if correct size.

			const IndexType nrFilters = this->getNrFilters();
			const Shape<unsigned int> &tensorShape = input.getShape();

			const IndexType out_dim = ((tensorShape[-1] - nrFilters) / this->stride_size[0]) + 1;

			Shape<unsigned int> output_shape = this->getShape();

			output = Tensor<DType>::zero(output_shape);

#pragma omp parallel for simd shared(filters)
			for (IndexType current_filter = 0; current_filter < nrFilters; current_filter++) {
				// TODO add matrix multiplication.
				Tensor<DType> filter = this->filters.getSubset({current_filter});

				for (IndexType x = 0; x < output_shape[0]; x++) {
					for (IndexType y = 0; y < output_shape[1]; y++) {

						//
						Tensor<DType> subset = input.getSubset(
							{(x * kernel_size[0] + kernel_size[0]), (y * kernel_size[1] + kernel_size[1])});

						const DType conv_value = (subset * filter).sum() + bias.getValue(current_filter);

						//
						output.getValue<float>({current_filter, y, x}) = conv_value;
					}
				}
			}
		}

		void initKernels(const Shape<IndexType> &buildShape) noexcept {
			RandomNormalInitializer<DType> init(0, 2);
			Shape<IndexType> kernelWeightShape = {{kernel_size[0], kernel_size[1], buildShape[-1]}};

			this->filters = init.get(kernelWeightShape);
		}

		void initbias(const Shape<IndexType> &buildShape) noexcept {
			RandomNormalInitializer<DType> init(0, 2);

			this->bias = init.get(Shape<IndexType>({buildShape[-1]}));
		}

	  protected:
		size_t getNrFilters() const noexcept { return this->filters.getShape()[-1]; }

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

		IndexType filter_count;
		Tensor<float> bias;
		Tensor<float> filters;

		std::array<uint32_t, 2> kernel_size;
		std::array<uint32_t, 2> stride_size;
	};
} // namespace Ritsu