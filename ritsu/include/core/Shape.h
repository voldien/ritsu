#pragma once
#include "../Math.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <istream>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	template <typename T> class Shape {
		static_assert(std::is_integral<T>::value, "Type must be a integral type.");

		using IndexType = T;
		static constexpr size_t IndexTypeSize = sizeof(IndexType);

	  public:
		Shape() = default;
		// TODO: initializer_list
		// Shape(std::initializer_list<IndexType> list) { this->dims = std::move(list); }
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

		auto &operator=(const std::vector<IndexType> &shape) {
			this->dims = shape;
			return *this;
		}

		auto &operator=(std::vector<IndexType> &&shape) {
			this->dims = std::move(shape);
			return *this;
		}

		// TODO: determine how
		friend Shape operator+(const Shape &shapeA, const Shape &shapeB) {
			// verify if operation is possible.
			std::vector<IndexType> size = shapeA.dims;
			for (size_t i = 0; i < size.size(); i++) {
				size[i] += shapeB.getAxisDimensions(i);
			}

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

		bool operator!=(const Shape<IndexType> &shape) const { return !(*this == shape); }

		Shape<IndexType> getSubShape(const size_t start, const size_t end) const {

			/*	*/
			typename std::vector<IndexType>::const_iterator first = this->dims.begin() + start;
			typename std::vector<IndexType>::const_iterator last =
				(this->dims.begin() + Math::mod<size_t>(end, this->dims.size())) + 1; // Inclusive.

			std::vector<IndexType> newVec(first, last);
			return Shape<IndexType>(newVec);
		}

		Shape<IndexType> getSubShape(const size_t start) const {
			return this->getSubShape(start, this->getNrDimensions() - 1);
		}

		// implicit
		operator std::vector<IndexType>() const { return this->dims; }
		// explicit conversion
		explicit operator const std::vector<T> &() const { return this->dims; }

		const Shape<IndexType> &reduce() const {
			// Remove 1 axis.
			for (size_t i = 0; i < this->getNrDimensions(); i++) {
				// TODO
			}

			return *this;
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

			// TODO: impl
			for (size_t i = 0; i < this->dims.size(); i++) {
			}
			return *this;
		}

		IndexType getNrElements() const { return Shape::computeNrElements<IndexType>(this->dims); }

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

		void append(const std::vector<IndexType> &additionalDims) {}

		template <typename U> static size_t computeIndex(const std::vector<U> &dim) {
			size_t totalSize = 1;

			for (size_t i = dim.size() - 1; i >= dim.size(); i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize;
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