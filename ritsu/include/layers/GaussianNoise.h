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
#include "Layer.h"
#include "Random.h"
#include <ctime>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class GuassianNoise : public Layer<float> {
	  public:
		GuassianNoise(const DType mean, const DType stddev, const std::string &name = "noise")
			: Layer(name), random(new RandomNormal<DType>(stddev, mean)) {}
		~GuassianNoise() override { delete this->random; }

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->applyNoise(tensor);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmp = tensor;
			this->applyNoise(tmp);
			return tmp;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->applyNoise(tensor);
			return tensor;
		}

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override {
			this->applyNoise(tensor);
			return tensor;
		}

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> tmp = tensor;
			this->applyNoise(tmp);
			return tmp;
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];
			this->shape = this->input->getShape();
		}

		void build(const Shape<IndexType> &buildShape) override { this->shape = buildShape; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		/*	No derivative.	*/
		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		/*	*/
		Layer<DType> *input{};
		std::vector<Layer<DType> *> outputs;

	  protected:
		void applyNoise(Tensor<float> &tensor) noexcept {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel for simd shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) += this->random->rand();
			}
		}

	  private:
		Random<DType> *random;
	};
} // namespace Ritsu