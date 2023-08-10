#pragma once
#include "../Math.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <istream>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

namespace Ritsu {

	template <typename T = unsigned int> class Shape {
	  public:
		Shape() = default;
		Shape(const std::vector<T> &shape) { this->dimensions = shape; }

		auto &operator=(const std::vector<T> &shape) {
			this->dimensions = shape;
			return *this;
		}

		T operator[](T location) const { return this->dimensions[location]; }
		T &operator[](T location) { return this->dimensions[location]; }

		bool operator==(const Shape &shape) const {
			if (&shape == this) {
				return true;
			}
			return shape.dimensions == this->dimensions;
		}
		bool operator!=(const Shape &shape) const {
			if (&shape != this) {
				return true;
			}

			return shape.dimensions != this->dimensions;
		}

		Shape flatten() const { return Shape({(T)Shape::computeNrElements(this->dimensions)}); }

		T getNrElements() const { return computeNrElements<T>(this->dimensions); }

		// TODO remove
		const std::vector<T> &getDims() const { return this->dimensions; }

		friend std::ostream &operator<<(std::ostream &os, const Shape &shape) {

			os << "[";
			for (int i = 0; i < shape.dimensions.size(); i++) {
				size_t index = i;
				T value = shape.dimensions[i];
				os << value;
				if (i < shape.dimensions.size() - 1) {
					os << ",";
				}
			}
			os << "]";

			return os;
		}

		void Reshape() {}

	  public:
		template <typename U> static U computeNrElements(const std::vector<U> &shape) {
			size_t totalSize = 1;
#pragma omp for simd
			for (size_t i = 0; i < shape.size(); i++) {
				totalSize *= shape[i];
			}
			return totalSize;
		}

	  private:
		std::vector<T> dimensions;
	};
} // namespace Ritsu