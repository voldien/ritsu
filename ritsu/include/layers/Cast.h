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

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename A, typename T> class Cast : public Layer<T> {
	  public:
		Cast(const std::string &name = "cast") : Layer<T>(name){};

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			tensor = this->createCastTensor(tensor);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override { return this->createCastTensor(tensor); }

		Tensor<T> &call(Tensor<T> &tensor, bool training) override {
			this->createCastTensor(tensor);
			return tensor;
		}

		Tensor<T> call(const Tensor<T> &tensor, bool training) override {
			Tensor<float> tmp = this->createCastTensor(tensor);
			return tmp;
		}

		void setInputs(const std::vector<Layer<T> *> &layers) override {
			this->input = layers[0];

			this->shape = this->input->getShape();
		}

		void setOutputs(const std::vector<Layer<T> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void build(const Shape<typename Layer<T>::IndexType> &shape) override { this->shape = shape; }

		std::vector<Layer<T> *> getInputs() const override { return {input}; }
		std::vector<Layer<T> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

		const std::type_info &getFromCastType() const noexcept { return typeid(A); }
		const std::type_info &getToCastType() const noexcept { return typeid(T); }

	  protected: /*	*/
		static inline Tensor<float> createCastTensor(const Tensor<float> &tensor) {
			Tensor<float> castTensor(tensor.getShape(), sizeof(T));

			return Cast::createCastTensorRef(castTensor);
		}

		static inline Tensor<float> &createCastTensorRef(Tensor<float> &tensor) {
			/*	*/
			tensor = reinterpret_cast<Tensor<float> &&>(std::move(tensor.cast<T>()));
			return tensor;
		}

	  private:
		Layer<T> *input;
		std::vector<Layer<T> *> outputs;
	};
} // namespace Ritsu