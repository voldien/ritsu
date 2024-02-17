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
#include "RitsuDef.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <forward_list>
#include <initializer_list>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Ritsu {

	template <typename V> class ShapePair {
	  public:
		constexpr ShapePair(const V start) : start0(start), end0(start) {}
		constexpr ShapePair(const V start, V end) : start0(start), end0(end) {}
		V start0;
		V end0;
	};

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
		Shape(const std::vector<IndexType> &shape) noexcept { this->dims = shape; }
		Shape(const Shape &shape) noexcept { this->dims = shape.dims; }
		Shape(Shape &&shape) noexcept { this->dims = std::move(shape.dims); }

		Shape &operator=(const Shape &shape) noexcept {
			this->dims = shape.dims;
			return *this;
		}

		Shape &operator=(Shape &&shape) noexcept {
			this->dims = std::move(shape.dims);
			return *this;
		}

		auto &operator=(const std::initializer_list<IndexType> &shape) noexcept {
			this->dims = shape;
			return *this;
		}

		auto &operator=(std::vector<IndexType> &&shape) noexcept {
			this->dims = std::move(shape);
			return *this;
		}

		auto &operator=(const std::vector<IndexType> &shape) noexcept {
			this->dims = shape;
			return *this;
		}

		friend Shape operator+(const Shape &shapeA, const Shape &shapeB) { return shapeA.append(shapeB); }

		friend Shape operator-(const Shape &shapeA, const Shape &shapeB) { return shapeA.erase(shapeB); }

		IndexType operator[](IndexType index) const noexcept { return this->getAxisDimensions(index); }
		IndexType &operator[](IndexType index) noexcept { return this->getAxisDimensions(index); }

		Shape<IndexType> operator()(IndexType start, IndexType end) const { return this->getSubShapeMem(start, end); }

		bool operator==(const Shape<IndexType> &shape) const noexcept {

			/*	Same address => equal.	*/
			if (static_cast<const void *>(&shape) == static_cast<const void *>(this)) {
				return true;
			}

			if (shape.getNrElements() != this->getNrElements()) {
				return false;
			}

			const IndexType minDim = Math::min(shape.getNrDimensions(), this->getNrDimensions());

			/*	Check all elements equal.	*/
			for (size_t i = 0; i < minDim; i++) {
				if (shape.dims[i] == this->dims[i]) {
					continue;
				}
				return false;
			}

			const IndexType maxDim = Math::max(shape.getNrDimensions(), this->getNrDimensions());

			/*	If equal size dim => same.	*/
			if (minDim == maxDim) {
				return true;
			}

			/*	Check additional dims are equal to 1.	TODO:cleanup.	*/
			for (IndexType i = minDim; i < maxDim; i++) {
				if (i < shape.getNrDimensions()) {
					if (shape.getAxisDimensions(i) != 1) {
						return false;
					}
				}
				if (i < this->getNrDimensions()) {
					if (getAxisDimensions(i) != 1) {
						return false;
					}
				}
			}
			return true;
		}

		bool operator==(const std::initializer_list<IndexType> &list) const noexcept {

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

		bool operator!=(const Shape<IndexType> &shape) const noexcept { return !(*this == shape); }

		Shape<IndexType> getSubShapeMem(const IndexType start, const IndexType end) const {
			/*	*/
			// TODO: validate request.

			/*	*/
			typename std::vector<IndexType>::const_iterator first = this->dims.begin() + start;

			typename std::vector<IndexType>::const_iterator last =
				(this->dims.begin() + Math::mod<int32_t>(end, this->dims.size())) + 1; // Inclusive.

			std::vector<IndexType> newVec(first, last);

			return Shape<IndexType>(newVec);
		}

		/**
		 * @brief Get the Sub Shape object
		 */
		Shape<IndexType> getSubShape(const int axis) const {
			/*	*/
			return this->getSubShapeMem(axis, this->getNrDimensions() - 1);
		}

		template <typename... Args> Shape<IndexType> getSubShape2(const Args &... args) const {

			auto res = {std::forward<Args>(args)...};
			//	const std::initializer_list<ShapePair<IndexType>> res_ = {
			//		ShapePair<IndexType>(std::forward<Args>(args))...};

			//	return getSubShape3(res_);
		}

		Shape<IndexType> getSubShape(const std::initializer_list<ShapePair<IndexType>> subaxisGroup) const {
			/*	*/
			if (subaxisGroup.size() > this->getNrDimensions()) {
				throw InvalidArgumentException("Sub shape has higher dimensions than shape");
			}

			Shape<IndexType> copy = *this;

			/*	*/
			for (size_t i = 0; i < subaxisGroup.size(); i++) {

				const ShapePair<IndexType> &pair = (*(subaxisGroup.begin() + i));

				const int sign_diff = static_cast<int>(pair.start0) - static_cast<int>(pair.end0);

				const int diff = Math::abs<int>(sign_diff);
				if (diff == 0) {
					copy[i] = 1;
				} else {
					copy[i] = diff + 1;
				}
			}

			return copy;
		}

		/**
		 * @brief implicit
		 *
		 * @return std::vector<IndexType>
		 */
		operator std::vector<IndexType>() const noexcept { return this->dims; }

		/**
		 * @brief
		 *
		 * @return const std::vector<T> &
		 */
		explicit operator const std::vector<IndexType> &() const noexcept { return this->dims; }

		/**
		 * @brief
		 *
		 * @return Shape<IndexType>&
		 */
		Shape<IndexType> &reduce() noexcept {

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

		/**
		 * @brief
		 */
		Shape<IndexType> flatten() const noexcept {
			return Shape({static_cast<IndexType>(Shape::computeNrElements<IndexType>(this->dims))});
		}

		/**
		 * @brief
		 */
		Shape<IndexType> &flatten() noexcept {
			*this = Shape({static_cast<IndexType>(Shape::computeNrElements<IndexType>(this->dims))});
			return *this;
		}

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

		inline IndexType getNrElements() const noexcept { return Shape::computeNrElements<IndexType>(this->dims); }

		IndexType getAxisDimensions(const int32_t index) const noexcept {
			return this->dims[Math::mod<int32_t>(index, this->dims.size())];
		}

		IndexType &getAxisDimensions(const int32_t index) noexcept {
			return this->dims[Math::mod<int32_t>(index, this->dims.size())];
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
			if (Shape<IndexType>::computeNrElements(newDims) == this->getNrElements()) {
				this->dims = newDims;

				// TODO determine if the data has to reshaped too.
				if (*this != newDims) {
					throw RuntimeException("Failed to reshape");
				}

			} else {
				// Failure
				throw InvalidArgumentException("Invalid Dimension");
			}
			return *this;
		}

		/**
		 * @brief
		 *
		 * @param additionalDims
		 * @param axis
		 * @return Shape<IndexType>&
		 */
		Shape<IndexType> &append(const Shape<IndexType> &additionalDims, const int axis = -1) {

			if (!canMerge(*this, additionalDims, axis)) {
				throw RuntimeException("Invalid");
			}
			const unsigned int axisMod = Math::mod<int>(axis, this->getNrDimensions());
			this->getAxisDimensions(axisMod) += additionalDims[axisMod];

			return *this;
		}

		Shape<IndexType> append(const Shape<IndexType> &additionalDims, const int axis = -1) const {

			if (!canMerge(*this, additionalDims, axis)) {
				throw RuntimeException("Invalid");
			}

			const unsigned int axisMod = Math::mod<int32_t>(axis, this->getNrDimensions());
			std::vector<IndexType> size = this->dims;

			size[axisMod] += additionalDims[axisMod];

			return size;
		}

		Shape<IndexType> &insert(int axis, const Shape<IndexType> &additionalDims) {

			this->dims.insert(std::begin(this->dims) + 1, additionalDims.dims.begin(), additionalDims.dims.end());
			return *this;
		}

		Shape<IndexType> &erase(const Shape<IndexType> &additionalDims, int axis = -1) {

			if (!canMerge(*this, additionalDims, axis)) {
				throw RuntimeException("Invalid");
			}

			const unsigned int axisMod = Math::mod<int32_t>(axis, this->getNrDimensions());
			this->getAxisDimensions(axisMod) -= additionalDims[axisMod];

			return *this;
		}

		Shape<IndexType> erase(const Shape<IndexType> &additionalDims, int axis = -1) const {

			if (!canMerge(*this, additionalDims, axis)) {
				throw RuntimeException("Invalid");
			}

			const unsigned int axisMod = Math::mod<int32_t>(axis, this->getNrDimensions());
			std::vector<IndexType> size = this->dims;

			size[axisMod] -= additionalDims[axisMod];

			return size;
		}

	  public: /*	Static methods.	*/
		static Shape flatten(const Shape &shape) noexcept {
			return Shape({(IndexType)Shape::computeNrElements(shape.dims)});
		}

		/**
		 * @brief Get the Index Memory Offset object
		 * Memory is row. thus the result is itself.
		 */
		static inline IndexType getIndexMemoryOffset(const Shape<IndexType> &shape, IndexType index,
													 unsigned int orderAxis = 0) noexcept {
			if (orderAxis == 0) {
				return index;
			}

			/*	*/
			const IndexType axisDim = shape.getAxisDimensions(orderAxis);
			const IndexType depthSlice = computeDepth(shape, orderAxis);
			/*	*/
			return (index % axisDim) * depthSlice + (index / axisDim);
		}

		// TODO: rename
		static bool canMerge(const Shape<IndexType> &shapeA, const Shape<IndexType> &shapeB, int axis) noexcept {
			// verify if operation is possible.
			if (shapeA.getNrDimensions() != shapeB.getNrDimensions()) {
				return false;
			}

			const unsigned int axisMod = Math::mod<int32_t>(axis, shapeB.getNrDimensions());
			for (size_t i = 0; i < shapeB.getNrDimensions(); i++) {
				if (axisMod == i) {
					continue;
				}
				if (shapeB.getAxisDimensions(i) != shapeA.getAxisDimensions(i)) {
					return false;
				}
			}

			return true;
		}

		template <typename U>
		static size_t computeIndex(const std::vector<U> &dim, const Shape<IndexType> &shape) noexcept {
			size_t totalSize = 0;

			for (long i = 0; i < shape.getNrDimensions(); i++) {
				long depth = 1;
				if (i > 0) {
					depth = Math::product(&shape.dims.data()[i], i);
				}

				if (i < dim.size()) {
					totalSize += depth * *(dim.begin() + i);
				}
			}
			return totalSize;
		}

		template <typename U>
		static size_t computeIndex(const std::initializer_list<U> &dim, const Shape<IndexType> &shape) noexcept {
			size_t totalSize = 0;

			for (long i = 0; i < shape.getNrDimensions(); i++) {
				long depth = 1;
				if (i > 0) {
					depth = Math::product(&shape.dims.data()[i], i);
				}

				if (i < dim.size()) {
					totalSize += depth * *(dim.begin() + i);
				}
			}
			return totalSize;
		}

		/**
		 * @brief
		 */
		static inline IndexType computeDepth(const Shape<IndexType> &shape, int depth) noexcept {
			if (depth > 0) {
				return Math::product<IndexType>(shape.dims.data(), depth);
			}
			return 1;
		}

		template <typename U> static inline U computeNrElements(const std::vector<U> &dims) noexcept {
			static_assert(std::is_integral<U>::value, "Type must be a integral type.");
			return Math::product(dims);
		}

	  private:
		std::vector<IndexType> dims;
		IndexType count;
		std::vector<IndexType> cacheDim;
	};
} // namespace Ritsu