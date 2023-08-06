#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
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
			this->resizeBuffer(dimensions, elementSize);
		}
		Tensor(uint8_t *buffer, size_t size, const std::vector<unsigned int> &dimensions) {
			this->buffer = buffer;
			this->dimensions = dimensions;
			this->NrElements = this->compute_number_elements(this->dimensions);
		}
		Tensor(const Tensor &other) {
			this->resizeBuffer(other.dimensions, other.DTypeSize);
			memcpy(this->buffer, other.buffer, other.getDatSize());
		}
		Tensor(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, nullptr);
			this->dimensions = other.dimensions;
			this->NrElements = other.NrElements;
		}
		~Tensor() {
			if (!this->isSubset) {
				free(this->buffer);
			}
			this->buffer = nullptr;
		}

		auto &operator=(const Tensor &other) {
			this->resizeBuffer(other.dimensions, other.DTypeSize);
			memcpy(this->buffer, other.buffer, other.getDatSize());

			return *this;
		}
		auto &operator=(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, nullptr);
			this->dimensions = other.dimensions;
			this->NrElements = other.NrElements;
			return *this;
		}

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

		auto &operator+(const Tensor &a) {
			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(a)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) + a.getValue<float>(n);
			}
			return *this;
		}

		template <typename T> auto &operator+(const T &v) {
			size_t nrElements = this->getNrElements();
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) + v;
			}
			return *this;
		}

		template <typename T> auto &operator-(const Tensor &tensor) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) - tensor.getValue<float>(n);
			}
			return *this;
		}

		template <typename T> auto &operator*(const Tensor &tensor) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) * tensor.getValue<float>(n);
			}
			return *this;
		}

		auto &operator*(const DType &v) {
			size_t nrElements = this->getNrElements();
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) * v;
			}
			return *this;
		}

		template <typename T> auto &operator/(const Tensor &tensor) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) / tensor.getValue<float>(n);
			}
			return *this;
		}

		// template <typename T> auto &operator*(T a) { return *this; }
		//
		// template <typename T> auto &operator/(T a) { return *this; }

		void assignInitValue(const DType &initValue) {
			for (size_t i = 0; i < this->getNrElements(); i++) {
				this->getValue<DType>(i) = initValue;
			}
		}

		template <typename T> auto getSubset(size_t start, size_t end) const {
			Tensor subset = T(static_cast<uint8_t *>(&this->buffer[start * DTypeSize]), end - start, this->dimensions);
			subset.isSubset = true;
			return subset;
		}

		Tensor &flatten() {
			/*	flatten dim.	*/
			this->dimensions = {(unsigned int)this->getNrElements()};
			return *this;
		}

		template <typename T> Tensor &append(T v) { return *this; }
		Tensor &append(Tensor &a) { return *this; }

		// TODO add std::cout istream

		float operator[](const std::vector<unsigned int> &location) const { return getValue<float>(location); }
		float &operator[](const std::vector<unsigned int> &location) { return getValue<float>(location); }

		void resizeBuffer(const std::vector<unsigned int> &dimensions, unsigned int elementSize) {
			size_t total_elements = this->compute_number_elements(dimensions);

			this->buffer = new uint8_t[total_elements * elementSize];
			this->dimensions = dimensions;
			this->NrElements = this->compute_number_elements(this->dimensions);
		}

		friend std::ostream &operator<<(std::ostream &os, Tensor &tensor) {
			size_t number_elements = tensor.getNrElements();
			for (int i = 0; i < number_elements; i++) {
				size_t index = i * 4;
				DType value = static_cast<DType>(tensor[{(unsigned int)i}]);
				os << value << ",";
			}

			return os;
		}

		const std::vector<unsigned int> &getShape() const { return this->dimensions; }

		inline size_t computeIndex(const std::vector<unsigned int> &dim) const {
			size_t totalSize = 1;

			for (size_t i = dim.size() - 1; i >= dim.size(); i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize;
		}
		template <typename T> inline const T *getRawData() const { return reinterpret_cast<T *>(this->buffer); }

		inline size_t getNrElements() const { return this->NrElements; }
		inline size_t getDatSize() const { return this->getNrElements() * DTypeSize; }

		static bool verifyShape(const Tensor &a, const Tensor &b) {
			if (a.getShape() != b.getShape()) {
				return false;
			}
			if (a.getNrElements() != b.getNrElements()) {
				return false;
			}
			// TODO add more
			return true;
		}

	  protected:
		inline size_t compute_number_elements(const std::vector<unsigned int> &dims) const {
			size_t totalSize = 1;
#pragma omp for simd
			for (size_t i = 0; i < dims.size(); i++) {
				totalSize *= dims[i];
			}
			return totalSize;
		}

	  private:
		size_t NrElements = 0;
		std::vector<unsigned int> dimensions;
		// TODO custom buffer for improved performance.
		union {
			uint8_t *buffer = nullptr;
			float *fbuffer;
		};
		bool isSubset;
	};
} // namespace Ritsu