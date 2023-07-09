#pragma once
#include <cmath>
#include <iostream>
#include <istream>
#include <omp.h>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace Ritsu {

	/**
	 * @brief Multi dimensional array
	 *
	 */
	/*template <class T = float> */ class Tensor {
	  public:
		using DType = float;
		const size_t DTypeSize = sizeof(DType);
		using IndexType = unsigned int;

	  public:
		Tensor() = default;

		/**
		 * @brief Construct a new Tensor object
		 *
		 * @param dimensions
		 * @param elementSize
		 */
		Tensor(const std::vector<unsigned int> &dimensions, unsigned int elementSize) {
			resizeBuffer(dimensions, elementSize);
		}
		Tensor(const void *buffer, size_t size) {}
		Tensor(const Tensor &other) {
			this->buffer = other.buffer;
			this->dimensions = other.dimensions;
		}
		Tensor(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, this->buffer);
			this->dimensions = other.dimensions;
		}
		~Tensor() = default;

		auto &operator=(const Tensor &other) {
			this->buffer = other.buffer;
			this->dimensions = other.dimensions;
			return *this;
		}
		auto &operator=(Tensor &&other) { return *this; }

		// operations of data.
		template <typename T> inline const auto &getValue(const std::vector<unsigned int> &location) const {
			const size_t index = this->computeIndex(location);

			const T *addr = reinterpret_cast<const T *>(&buffer[index * 4]);
			return *addr;
		}

		template <typename T> inline auto &getValue(const std::vector<unsigned int> &location) {
			const size_t index = this->computeIndex(location);

			T *addr = reinterpret_cast<T *>(&buffer[index * 4]);
			return *addr;
		}

		template <typename T> inline auto &getValue(size_t index) {
			T *addr = reinterpret_cast<T *>(&buffer[index * 4]);
			return *addr;
		}

		template <typename T> inline const auto &getValue(size_t index) const {
			const T *addr = reinterpret_cast<const T *>(&buffer[index * 4]);
			return *addr;
		}

		template <typename T> auto getSubset(size_t start, size_t end) const { return T(); }

		auto &operator+(const Tensor &a) {
			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(a)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) + a.getValue<float>(n);
			}
			return *this;
		}

		template <typename T> auto &operator-(Tensor &a) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(a)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) - a.getValue<float>(n);
			}
			return *this;
		}

		template <typename T> auto &operator*(Tensor &a) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(a)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) * a.getValue<float>(n);
			}
			return *this;
		}

		template <typename T> auto &operator/(Tensor &a) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(a)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) / a.getValue<float>(n);
			}
			return *this;
		}

		// template <typename T> auto &operator*(T a) { return *this; }
		//
		// template <typename T> auto &operator/(T a) { return *this; }

		Tensor &flatten() { return *this; }

		template <typename T> Tensor &append(T v) { return *this; }
		Tensor &append(Tensor &a) { return *this; }

		// TODO add std::cout istream

		float operator[](const std::vector<unsigned int> &location) const { return getValue<float>(location); }
		float &operator[](const std::vector<unsigned int> &location) { return getValue<float>(location); }

		void resizeBuffer(const std::vector<unsigned int> &dimensions, unsigned int elementSize) {
			size_t totalSize = 1;
			for (size_t i = 0; i < dimensions.size(); i++) {
				totalSize *= dimensions[i];
			}
			buffer.resize(totalSize * elementSize);
			this->dimensions = dimensions;
		}

		friend std::ostream &operator<<(std::ostream &os, Tensor &tensor) {
			size_t number_elements = tensor.buffer.size() / tensor.DTypeSize;
			for (int i = 0; i < number_elements; i++) {
				size_t index = i * 4;
				DType value = static_cast<DType>(tensor[{(unsigned int)i}]);
				os << value << ",";
			}

			return os;
		}

		const std::vector<unsigned int> &getNrDimension() const { return this->dimensions; }

		inline size_t computeIndex(const std::vector<unsigned int> &dim) const {
			size_t totalSize = 1;

			for (size_t i = dim.size() - 1; i >= dim.size(); i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize;
		}

		inline size_t getNrElements() const {
			size_t totalSize = 1;
#pragma omp for simd
			for (size_t i = 0; i < this->dimensions.size(); i++) {
				totalSize *= dimensions[i];
			}
			return totalSize;
		}

		static bool verifyShape(const Tensor &a, const Tensor &b) {
			if (a.getNrDimension() != b.getNrDimension()) {
				return false;
			}
			// TODO add more
			return true;
		}

	  private:
		std::vector<unsigned int> dimensions;
		// TODO custom buffer for improved performance.
		std::vector<uint8_t> buffer;
	};
} // namespace Ritsu