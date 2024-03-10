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
#include "Object.h"
#include "Tensor.h"
#include <functional>
#include <iostream>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	// TODO add template.
	// template <typename T>
	class Loss : public Object {
	  public:
		// static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
		//			  "Must be a decimal type(float/double/half) or integer.");
		//
		// using IndexType = unsigned int;
		// static constexpr size_t IndexTypeSize = sizeof(IndexType);
		// using DType = T;
		// const size_t DTypeSize = sizeof(DType);

	  public:
		using LossFunction = void (*)(const Tensor<float> &evaluated_pre_true, const Tensor<float> &expected_pred,
									  Tensor<float> &output_result);

		Loss() : Object("loss") {}
		//	template <typename T>
		Loss(LossFunction lambda, const std::string &name = "loss") noexcept : Object(name) {
			this->loss_function = lambda;
			/*	Cache buffer.	*/ // TODO:
		}

		// virtual Tensor<float> &computeLoss(const Tensor<float> &inputX0_true, const Tensor<float> &inputX1_pred)  =
		// 0;

		virtual Tensor<float> computeLoss(const Tensor<float> &inputX0_true, const Tensor<float> &inputX1_pred) const {

			Tensor<float> batchLossResult;

			/*	*/
			if (!Tensor<float>::verifyShape(inputX0_true, inputX1_pred)) {
				std::cerr << "Loss - Bad Shape " << inputX0_true.getShape() << " not equal " << inputX1_pred.getShape()
						  << std::endl;
			}

			this->loss_function(inputX0_true, inputX1_pred, batchLossResult);

			return batchLossResult;
		}

		virtual Tensor<float> operator()(const Tensor<float> &inputX0_true, const Tensor<float> &inputX1_pred) {
			return this->computeLoss(inputX0_true, inputX1_pred);
		}

	  private:
		LossFunction loss_function;
	};

	class CategoricalCrossentropy : public Loss {
	  public:
		// static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
		//			  "Must be a decimal type(float/double/half) or integer.");
		//
		// using IndexType = unsigned int;
		// static constexpr size_t IndexTypeSize = sizeof(IndexType);
		// using DType = T;
		// const size_t DTypeSize = sizeof(DType);

	  public:
		// CategoricalCrossentropy(const std::string name = "categorical_crossentropy") : Loss("loss") {}
		////	template <typename T>
		// CategoricalCrossentropy(LossFunction lambda, const std::string &name = "loss") noexcept : Object(name) {
		//	this->loss_function = lambda;
		//	/*	Cache buffer.	*/ // TODO:
		//}
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
		const int batchIndex = -1;
		output_result = output_result.mean(batchIndex);
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
		const int batchIndex = -1;
		output_result = output_result.mean(batchIndex);
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
		const int batchIndex = -1;
		output_result = output_result.mean(batchIndex);
	}

	/**
	 * @brief
	 */
	static void loss_binary_cross_entropy(const Tensor<float> &evaluated_pre, const Tensor<float> &expected,
										  Tensor<float> &output) {

		output = std::move(expected * Tensor<float>::log10(evaluated_pre) * -1.0f);

		const int batchIndex = -1;
		output = output.mean(batchIndex);
	}

	/**
	 * @brief
	 */
	static void loss_cross_entropy(const Tensor<float> &evaluated_pre, const Tensor<float> &expected,
								   Tensor<float> &output) {

		Tensor<float> logY = Tensor<float>::log10(evaluated_pre);

		// TODO add support for primitve
		output = std::move(-(expected * logY));
		const int batchIndex = -1;
		output = output.sum(batchIndex);
	}

	// TODO convert to one shot vector.
	/**
	 * @brief
	 */
	static void loss_categorial_crossentropy(const Tensor<float> &evaluated_pre, const Tensor<float> &expected_target,
											 Tensor<float> &output) { // axis = -1
		output = evaluated_pre;
		output.clip(1e-7, 1 - 1e-7);

		output = expected_target * -Tensor<float>::log10(output);

		const int batchIndex = -1;
		output = output.sum(batchIndex);
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

		return loss_categorial_crossentropy(evaluated_pre, expected_one_shot, output);
	}
} // namespace Ritsu
