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
#include "../Math.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	template <typename T> class Shape {
		static_assert(std::is_integral<T>::value, "Type must be a integral type.");

		using IndexType = T;
		static constexpr unsigned int IndexTypeSize = sizeof(IndexType);

	  public:
		Shape() = default;
		// TODO: initializer_list
		// Shape(const std::initializer_list<IndexType>& list) { this->dims = std::move(list); }
		Shape(const std::vector<IndexType> &shape) { this->dims = shape; }
		Shape(const Shape &shape) { this->dims = shape.dims; }
		Shape(Shape &&shape) { this->dims = std::move(shape.dims); }

		Shape &operator=(const Shape &shape) {
			this->dims = shape.dims;
			return *this;
		}

		Shape &operator=(Shape &&shape) {
			this->dims = std::move(shape.dims);
			return *this;
		}

		auto &operator=(const std::initializer_list<IndexType> &shape) {
			this->dims = shape;
			return *this;
		}

		auto &operator=(std::vector<IndexType> &&shape) {
			this->dims = std::move(shape);
			return *this;
		}

		auto &operator=(const std::vector<IndexType> &shape) {
			this->dims = shape;
			return *this;
		}

		// TODO: determine how
		friend Shape operator+(const Shape &shapeA, const Shape &shapeB) {
			// verify if operation is possible.
			std::vector<IndexType> size = shapeA.dims;

			size[size.size() - 1] += shapeB[shapeB.getNrDimensions() - 1];

			return size;
		}

		// TODO: determine how
		friend Shape operator-(const Shape &shapeA, const Shape &shapeB) {
			// verify if operation is possible.
			return {};
		}

		IndexType operator[](IndexType index) const { return this->getAxisDimensions(index); }
		IndexType &operator[](IndexType index) { return this->getAxisDimensions(index); }

		Shape<IndexType> operator()(IndexType start, IndexType end) const { return this->getSubShape(start, end); }

		bool operator==(const Shape<IndexType> &shape) const {

			/*	Same address => equal.	*/
			if (static_cast<const void *>(&shape) == static_cast<const void *>(this)) {
				return true;
			}

			/*	If not same number dims => not the same.	*/
			if (shape.getNrDimensions() != this->getNrDimensions()) {
				return false;
			}

			/*	Check all elements equal.	*/
			for (size_t i = 0; i < shape.getNrDimensions(); i++) {
				if (shape.dims[i] == this->dims[i]) {
					continue;
				}
				return false;
			}

			return true;
		}

		bool operator==(const std::initializer_list<IndexType> &list) const {

			/*	If not same number dims => not the same.	*/
			if (list.size() != this->getNrDimensions()) {
				return false;
			}

			/*	Check all elements equal.	*/
			for (size_t i = 0; i < list.size(); i++) {
				if (*(list.begin() + i) == this->dims[i]) {
					continue;
				}
				return false;
			}

			return true;
		}

		bool operator!=(const Shape<IndexType> &shape) const { return !(*this == shape); }

		Shape<IndexType> getSubShape(const size_t start, const size_t end) const {

			/*	*/
			typename std::vector<IndexType>::const_iterator first = this->dims.begin() + start;
			typename std::vector<IndexType>::const_iterator last =
				(this->dims.begin() + Math::mod<size_t>(end, this->dims.size())) + 1; // Inclusive.

			std::vector<IndexType> newVec(first, last);
			return Shape<IndexType>(newVec);
		}

		Shape<IndexType> getSubShape(const size_t axis) const {
			return this->getSubShape(axis, this->getNrDimensions() - 1);
		}

		// implicit
		operator std::vector<IndexType>() const { return this->dims; }
		// explicit conversion
		explicit operator const std::vector<T> &() const { return this->dims; }

		Shape<IndexType> &reduce() noexcept {
			// Remove 1 axis.

			for (size_t i = 0; i < this->getNrDimensions(); i++) {

				if (this->dims[i] == 1 || this->dims[i] == 0) {
					this->dims.erase(std::next(this->dims.begin(), i));
					i = -1; /*	reset index counter.	*/
				}
			}

			return *this;
		}

		Shape<IndexType> reduce() const noexcept {
			// Remove 1 axis.
			std::vector<IndexType> tmp = this->dims;
			for (size_t i = 0; i < tmp.size(); i++) {
				if (tmp[i] == 1 || tmp[i] == 0) {
					tmp.erase(tmp.begin() + i);
					i = -1; /*	reset index counter.	*/
				}
			}

			return tmp;
		}

		Shape<IndexType> flatten() const {
			return Shape({static_cast<IndexType>(Shape::computeNrElements<IndexType>(this->dims))});
		}

		Shape<IndexType> &flatten() {
			*this = Shape({static_cast<IndexType>(Shape::computeNrElements<IndexType>(this->dims))});
			return *this;
		}
		static Shape flatten(const Shape &shape) { return Shape({(IndexType)Shape::computeNrElements(shape.dims)}); }

		Shape<IndexType> &transpose() noexcept {

			if (this->getNrDimensions() == 1) {
				*this = {1, this->getNrElements()};

			} else if (this->getNrDimensions() == 2) {
				for (size_t x = 0; x < this->getAxisDimensions(0); x++) {
					for (size_t y = 0; y < this->getAxisDimensions(1); y++) {
						std::swap(dims[(x - 1) * this->getAxisDimensions(0) + y - 1],
								  dims[(x - 1) * this->getAxisDimensions(0) + y - 1]);
					}
				}
			} else {
				/*	*/
			}
			// TODO: impl

			return *this;
		}

		Shape<IndexType> transpose() const noexcept {
			Shape<IndexType> tmp = *this;

			return tmp;
		}

		IndexType getNrElements() const noexcept { return Shape::computeNrElements<IndexType>(this->dims); }

		IndexType getAxisDimensions(const uint32_t index) const noexcept {
			return this->dims[Math::mod<size_t>(index, this->dims.size())];
		}

		IndexType &getAxisDimensions(const uint32_t index) noexcept {
			return this->dims[Math::mod<size_t>(index, this->dims.size())];
		}

		IndexType getNrDimensions() const noexcept { return this->dims.size(); }

		friend std::ostream &operator<<(std::ostream &stream, const Shape &shape) {

			stream << "[";
			for (int i = 0; i < shape.dims.size(); i++) {

				const size_t index = i;
				const IndexType value = shape.dims[i];
				stream << value;
				if (i < shape.dims.size() - 1) {
					stream << ",";
				}
			}
			stream << "]";

			return stream;
		}

		// TODO add template to allow multiple of primtive types.
		Shape<IndexType> &reshape(const std::vector<IndexType> &newDims) {
			/*	Only allow reshape if both dims have the same number of elements.	*/
			if (this->computeNrElements(newDims) == this->getNrElements()) {
				this->dims = newDims;

				// TODO determine if the data has to reshaped too.
				if (*this != newDims) {
					throw std::runtime_error("Failed to reshape");
				}
			} else {
				// Failure
				throw std::invalid_argument("Invalid Dimension");
			}
			return *this;
		}

		Shape<IndexType> &append(const Shape<IndexType> &additionalDims, int axis = 0) {

			// verify if operation is possible.

			this->getAxisDimensions(this->getNrDimensions() - 1) +=
				additionalDims[additionalDims.getNrDimensions() - 1];

			return *this;
		}

		Shape<IndexType> append(const Shape<IndexType> &additionalDims, int axis = 0) const {
			// verify if operation is possible.
			std::vector<IndexType> size = this->dims;

			size[size.size() - 1] += additionalDims[additionalDims.getNrDimensions() - 1];

			return *this;
		}

		template <typename U> static size_t computeIndex(const std::vector<U> &dim) {
			size_t totalSize = 1;

			for (long i = dim.size() - 1; i >= 1; i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize - 1;
		}

		template <typename U> static size_t computeIndex(const std::initializer_list<U> &dim) {
			size_t totalSize = 1;

			for (long i = dim.size() - 1; i >= 1; i--) {
				totalSize *= *(dim.begin() + i);
			}
			totalSize += *(dim.begin());
			return totalSize - 1;
		}

	  public:
		template <typename U> static U computeNrElements(const std::vector<U> &dims) noexcept {
			static_assert(std::is_integral<U>::value, "Type must be a integral type.");
			return Math::product(dims);
		}

	  private:
		std::vector<IndexType> dims;
	};
} // namespace Ritsu