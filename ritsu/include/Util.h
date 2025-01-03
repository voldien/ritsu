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
#include "Random.h"
#include "Tensor.h"
namespace Ritsu {

	template <typename T> void shuffle_data(Tensor<T> &data, const int axis = 0, const size_t seed = 0) {

		const size_t count = data.getShape()[0];
		const Shape<typename Tensor<T>::IndexType> shape = data.getShape().getSubShape(1);

		RandomUniform<float> randomGenerator(0, 1, seed);

		for (size_t i = 0; i < count / 2; i++) {

			const typename Tensor<T>::IndexType swap0 = std::floor(randomGenerator.rand() * count);
			const typename Tensor<T>::IndexType swap1 = std::floor(randomGenerator.rand() * count);

			auto A =
				std::move(data.getSubset(swap0 * shape.getNrElements(), (swap0 + 1) * shape.getNrElements(), shape));
			auto B =
				std::move(data.getSubset(swap1 * shape.getNrElements(), (swap1 + 1) * shape.getNrElements(), shape));

			std::swap(A, B);
		}
	}

	// Split
	template <typename T>
	std::tuple<Tensor<T>, Tensor<T>> split_dataset(const Tensor<T> &dataset, const float left_side = 0.5f,
												   bool shuffle = false, int seed = 0, const bool parent = true) {
		const size_t left_size_count = dataset.getShape()[0] * left_side;
		const size_t right_size_count = dataset.getShape()[0] - left_size_count;

		assert(left_size_count + right_size_count == dataset.getShape()[0]);

		if (left_side >= 1) {
			return {dataset, {}};
		}

		Shape<typename Tensor<T>::IndexType> leftShape = dataset.getShape();
		leftShape[0] = left_size_count;

		Shape<typename Tensor<T>::IndexType> rightShape = dataset.getShape();
		rightShape[0] = right_size_count;

		Tensor<T> leftSplit;
		Tensor<T> rightSplit;

		if (parent) {
			leftSplit = dataset.getSubset(0, leftShape.getNrElements(), leftShape);
			rightSplit = dataset.getSubset(leftShape.getNrElements(),
										   leftShape.getNrElements() + rightShape.getNrElements(), rightShape);
		} else {
			//		Tensor<T>(dataset.getRawData(), leftShape.getNrElements() * dataset.getElementSize(), leftShape );
		}

		if (shuffle) {
			shuffle_data(leftSplit, 0, seed);
			shuffle_data(rightSplit, 0, seed);
		}

		return {leftSplit, rightSplit};
	}
} // namespace Ritsu