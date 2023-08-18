#pragma once
#include "../Math.h"
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
	template <typename T = unsigned int> class Shape {
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

		T operator[](T index) const { return this->getElementPerDimension(index); }
		T &operator[](T index) { return this->dims[index]; }

		bool operator==(const Shape &shape) const {
			if (&shape == this) {
				return true;
			}
			return shape.dims == this->dims;
		}

		bool operator!=(const Shape &shape) const {
			if (&shape != this) {
				return true;
			}

			return shape.dims != this->dims;
		}

		// implicit
		operator std::vector<T>() const { return this->dims; }
		// explicit conversion
		explicit operator const std::vector<T> &() const { return this->dims; }

		Shape flatten() const { return Shape({static_cast<T>(Shape::computeNrElements<T>(this->dims))}); }

		T getNrElements() const { return computeNrElements<T>(this->dims); }
		T getElementPerDimension(const uint32_t index) const { return this->dims[index]; }
		T getNrDimensions() const { return this->dims.size(); }

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

		void reshape(const std::vector<T> &newDims) {
			if (this->computeNrElements(newDims) == this->getNrElements()) {
				this->dims = newDims;
			} else {
				// Failure
				throw std::invalid_argument("Invalid Dimension");
			}
		}

	  public:
		Shape flatten(const Shape &shape) const { return Shape({(T)Shape::computeNrElements(shape.dims)}); }

		template <typename U> static U computeNrElements(const std::vector<U> &dims) {
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