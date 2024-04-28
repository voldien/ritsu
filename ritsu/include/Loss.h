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
#include "Activations.h"
#include "Object.h"
#include "Tensor.h"
#include <iostream>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	template <typename T> class Loss : public Object {
	  public:
		static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

	  public:
		using LossFunction = void (*)(const Tensor<DType> &evaluated_pre_true, const Tensor<DType> &expected_pred,
									  Tensor<DType> &output_result);

		Loss() : Object("loss") {}

		Loss(LossFunction lambda, const std::string &name = "loss") noexcept : Object(name) {
			this->loss_function = lambda;
		}

		virtual Tensor<DType> computeLoss(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) const {

			Tensor<DType> batchLossResult;

			/*	*/
			if (!Tensor<DType>::verifyShape(inputX0_true, inputX1_pred)) {
				std::cerr << "Loss - Bad Shape " << inputX0_true.getShape() << " not equal " << inputX1_pred.getShape()
						  << std::endl;
			}

			this->loss_function(inputX0_true, inputX1_pred, batchLossResult);

			return batchLossResult;
		}

		virtual Tensor<DType> derivative(const Tensor<DType> &inputX0_true,
										 const Tensor<DType> &inputX1_pred) const = 0;

		virtual Tensor<DType> operator()(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) {
			return this->computeLoss(inputX0_true, inputX1_pred);
		}

	  private:
		LossFunction loss_function;
	};

	/**
	 * @brief
	 */
	static void loss_error(const Tensor<float> &evaluated_pre, const Tensor<float> &expected,
						   Tensor<float> &output_result) {

		/*	(A - B)^2	*/
		output_result = evaluated_pre - expected;

		if (output_result.getShape()[0] == 1) {
			return;
		}

		/*	Mean for each batch index.	*/
		const int batchIndex = 0;
		output_result = output_result.mean(batchIndex);
	}

	/**
	 * @brief
	 */
	static void loss_categorical_crossentropy(const Tensor<float> &evaluated_pre, const Tensor<float> &expected_true,
											  Tensor<float> &output) { // axis = -1

		Tensor<float> tmp_output = evaluated_pre;

		output = -(expected_true * Tensor<float>::log10(tmp_output));

		const int batchIndex = 0;
		output = output.mean(batchIndex);
	}

	/**
	 * @brief
	 */
	static void sparse_categorical_crossentropy(const Tensor<float> &evaluated_pre, const Tensor<float> &expected,
												Tensor<float> &output) { // axis = -1

		const Tensor<float> expected_one_shot = Tensor<float>::zero(evaluated_pre.getShape());

		// expected_one_shot.getValue<float>((uint32_t)expected.getValue<float>(0)) = 1;

		/*	*/
		if (!Tensor<float>::verifyShape(evaluated_pre, expected_one_shot)) {
			std::cerr << "Sparse Categorical Crossentropy - Bad Shape " << expected_one_shot.getShape() << " not equal "
					  << evaluated_pre.getShape() << std::endl;
		}

		return loss_categorical_crossentropy(evaluated_pre, expected_one_shot, output);
	}

	class MeanSquareError : public Loss<float> {
	  public:
		MeanSquareError(const std::string name = "mse") : Loss(loss_mse, name) {}

		Tensor<DType> derivative(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) const override {
			Tensor<float> output_result = (inputX0_true - inputX1_pred) * -2;
			/*	Mean for each batch index.	*/
			const int batchIndex = 0;
			output_result = output_result.mean(batchIndex);
			return output_result;
		}

		/**
		 * @brief
		 */
		static void loss_mse(const Tensor<float> &evaluated_pre_true, const Tensor<float> &expected_pred,
							 Tensor<float> &output_result) {

			/*	(A - B)^2	*/
			output_result = evaluated_pre_true - expected_pred;
			output_result = output_result * output_result;

			/*	Mean for each batch index.	*/
			const int batchIndex = 0;
			output_result = output_result.mean(batchIndex);
		}
	};

	class MeanAbsoluterror : public Loss<float> {
	  public:
		MeanAbsoluterror(const std::string name = "mse") : Loss(loss_msa, name) {}
		Tensor<DType> derivative(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) const override {

			const DType cons = static_cast<DType>(1.0 / -2);
			Tensor<float> output_result =
				Tensor<float>::abs(inputX0_true - inputX1_pred) * static_cast<DType>(1.0 / -2);

			/*	Mean for each batch index.	*/
			const int batchIndex = 0;
			output_result = output_result.mean(batchIndex);
			return output_result;
		}

		/**
		 * @brief
		 */
		static void loss_msa(const Tensor<float> &evaluated_pre, const Tensor<float> &expected,
							 Tensor<float> &output_result) {

			/*	(A - B)	*/
			output_result = evaluated_pre - expected;
			output_result = output_result * output_result;

			/*	*/
			output_result = Tensor<float>::abs(output_result);

			/*	*/
			const int batchIndex = 0;
			output_result = output_result.mean(batchIndex);
		}
	};

	class BinaryCrossEntropy : public Loss<float> {
	  public:
		BinaryCrossEntropy(const std::string name = "binarycross") : Loss(loss_binary_cross_entropy, name) {}

		Tensor<DType> derivative(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) const override {
			Tensor<float> output_result = (inputX0_true - inputX1_pred) * -2;
			/*	Mean for each batch index.	*/
			const int batchIndex = 0;
			output_result = output_result.mean(batchIndex);
			return output_result;
		}

		static void loss_binary_cross_entropy(const Tensor<float> &evaluated_pre, const Tensor<float> &expected,
											  Tensor<float> &output) {

			output = std::move(expected * Tensor<float>::log10(evaluated_pre) * -1.0f);

			const int batchIndex = 0;
			output = output.mean(batchIndex);
		}
	};

	class CategoricalCrossentropy : public Loss<float> {
	  public:
		CategoricalCrossentropy(bool from_logits = false, const std::string name = "categorical_crossentropy")
			: Loss(Ritsu::loss_categorical_crossentropy, name), from_logits(from_logits) {}

		Tensor<DType> computeLoss(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) const override {
			Tensor<DType> batchLossResult = inputX1_pred;

			if (this->from_logits) {
				batchLossResult = Ritsu::softMax(batchLossResult);
			} else {
				batchLossResult = batchLossResult / batchLossResult.sum(0);
				batchLossResult.clip(1e-7, 1 - 1e-7);
			}

			return Loss::computeLoss(inputX0_true, batchLossResult);
		}

		Tensor<DType> derivative(const Tensor<DType> &inputX0_true, const Tensor<DType> &inputX1_pred) const override {
			Tensor<DType> output_result = (inputX0_true - inputX1_pred) * -2;
			/*	Mean for each batch index.	*/
			const int batchIndex = 0;
			output_result = output_result.mean(batchIndex);
			return output_result;
		}

	  private:
		bool from_logits;
	};

} // namespace Ritsu
