#pragma once
#include "../Math.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
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

		using IndexType = unsigned int;
		static constexpr size_t IndexTypeSize = sizeof(IndexType);

	  public:
		Shape() = default;
		Shape(const std::vector<T> &shape) { this->dims = shape; }
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

		auto &operator=(const std::vector<T> &shape) {
			this->dims = shape;
			return *this;
		}

		auto &operator=(std::vector<T> &&shape) {
			this->dims = shape;
			return *this;
		}

		friend Shape<IndexType> operator+(const Shape<IndexType> &shapeA, const Shape<IndexType> &shapeB) {
			// verify if operation is possible.

			return {};
		}

		friend Shape<IndexType> operator-(const Shape<IndexType> &shapeA, const Shape<IndexType> &shapeB) {
			// verify if operation is possible.
			return {};
		}

		T operator[](T index) const { return this->getAxisDimensions(index); }
		T &operator[](T index) { return this->dims[index]; }

		Shape<IndexType> operator()(T start, T end) const { return this->getSubShape(start, end); }

		bool operator==(const Shape &shape) const {

			/*	Same address => equal.	*/
			if (&shape == this) {
				return true;
			}

			/*	If not same number dims => not the same.	*/
			if (shape.getNrDimensions() != this->getNrDimensions()) {
				return false;
			}

			/*	Check all elements equal.	*/
			for (size_t i = 0; i < shape.getNrDimensions(); i++) {
				if (shape.dims[i] != this->dims[i]) {
					return false;
				}
			}

			return true;
		}

		bool operator!=(const Shape &shape) const { return !(*this == shape); }

		Shape<IndexType> getSubShape(const size_t start, const size_t end) const {
			std::vector<IndexType>::const_iterator first = this->dims.begin() + start;
			std::vector<IndexType>::const_iterator last = this->dims.begin() + end;
			std::vector<IndexType> newVec(first, last);
			return Shape<IndexType>(newVec);
		}

		// implicit
		operator std::vector<T>() const { return this->dims; }
		// explicit conversion
		explicit operator const std::vector<T> &() const { return this->dims; }

		Shape flatten() const { return Shape({static_cast<T>(Shape::computeNrElements<T>(this->dims))}); }

		T getNrElements() const { return computeNrElements<T>(this->dims); }

		T getAxisDimensions(const uint32_t index) const noexcept {
			return this->dims[Math::mod<size_t>(index, this->dims.size())];
		}

		T getNrDimensions() const noexcept { return this->dims.size(); }

		friend std::ostream &operator<<(std::ostream &stream, const Shape &shape) {

			stream << "[";
			for (int i = 0; i < shape.dims.size(); i++) {
				size_t index = i;
				T value = shape.dims[i];
				stream << value;
				if (i < shape.dims.size() - 1) {
					stream << ",";
				}
			}
			stream << "]";

			return stream;
		}

		// TODO add template to allow multiple of primtive types.
		void reshape(const std::vector<T> &newDims) {
			/*	Only allow reshape if both dims have the same number of elements.	*/
			if (this->computeNrElements(newDims) == this->getNrElements()) {
				this->dims = newDims;
				// TODO determine if the data has to reshaped too.
			} else {
				// Failure
				throw std::invalid_argument("Invalid Dimension");
			}
		}

		static size_t computeIndex(const std::vector<IndexType> &dim) {
			size_t totalSize = 1;

			for (size_t i = dim.size() - 1; i >= dim.size(); i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize;
		}

	  public:
		Shape flatten(const Shape &shape) const { return Shape({(T)Shape::computeNrElements(shape.dims)}); }

		template <typename U> static U computeNrElements(const std::vector<U> &dims) noexcept {
			static_assert(std::is_integral<U>::value, "Type must be a integral type.");

			size_t totalSize = 1;

			/*	*/
#pragma omp for simd
			for (size_t i = 0; i < dims.size(); i++) {
				totalSize *= dims[i];
			}
			/*	*/
			return totalSize;
		}

	  private:
		std::vector<T> dims;
	};
} // namespace Ritsu