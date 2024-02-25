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
		using LossFunction = void (*)(const Tensor<float> &evoluated_true, const Tensor<float> &expected_pred,
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

	/**
	 * @brief
	 */
	static void loss_error(const Tensor<float> &evoluated, const Tensor<float> &expected,
						   Tensor<float> &output_result) {

		/*	(A - B)^2	*/
		output_result = evoluated - expected;

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
	static void loss_mse(const Tensor<float> &evoluated_true, const Tensor<float> &expected_pred,
						 Tensor<float> &output_result) {

		/*	(A - B)^2	*/
		output_result = evoluated_true - expected_pred;
		output_result = output_result * output_result;

		/*	Mean for each batch index.	*/
		const int batchIndex = -1;
		output_result = output_result.mean(batchIndex);
	}

	/**
	 * @brief
	 */
	static void loss_msa(const Tensor<float> &evoluated, const Tensor<float> &expected, Tensor<float> &output_result) {

		/*	(A - B)	*/
		output_result = evoluated - expected;
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
	static void loss_binary_cross_entropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
										  Tensor<float> &output) {

		output = std::move(expected * Tensor<float>::log10(evoluated) * -1.0f);

		const int batchIndex = -1;
		output = output.mean(batchIndex);
	}

	/**
	 * @brief
	 */
	static void loss_cross_entropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
								   Tensor<float> &output) {

		Tensor<float> A = Tensor<float>::log10(expected);

		// TODO add support for primitve
		Tensor<float> one(evoluated.getShape());
		output = std::move(-expected * A + (1.0f - expected) * A);

		const int batchIndex = -1;
		output = output.mean(batchIndex);
	}

	// TODO convert to one shot vector.
	/**
	 * @brief
	 */
	static void loss_cross_catagorial_entropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
											  Tensor<float> &output) {

		output = std::move(expected * Tensor<float>::log10(evoluated) * -1.0f);

		/*Tensor<float> A = inputA * log(inputB);*/
	}

	/**
	 * @brief
	 */
	static void sparse_categorical_crossentropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
												Tensor<float> &output) {

		Tensor<float> expected_one_shot = Tensor<float>::zero(evoluated.getShape());

		// expected_one_shot.getValue<float>((uint32_t)expected.getValue<float>(0)) = 1;

		/*	*/
		if (!Tensor<float>::verifyShape(evoluated, expected_one_shot)) {
			std::cerr << "Sparse Categorical Crossentropy - Bad Shape " << expected_one_shot.getShape() << " not equal "
					  << evoluated.getShape() << std::endl;
		}

		return loss_cross_catagorial_entropy(evoluated, expected_one_shot, output);
	}
} // namespace Ritsu
