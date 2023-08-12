#pragma once
#include "core/Shape.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <istream>
#include <memory>
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
		// static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
		//			  "Must be a decimal type(float/double/half) or integer.");

	  public:
		using DType = float;
		static constexpr size_t DTypeSize = sizeof(DType);
		using IndexType = unsigned int;
		static constexpr size_t IndexTypeSize = sizeof(IndexType);

	  public:
		Tensor() = default;

		/**
		 * @brief Construct a new Tensor object
		 *
		 * @param dimensions
		 * @param elementSize
		 */
		// TODO remove element size, replace with the generic type size.
		Tensor(const std::vector<IndexType> &dimensions, size_t elementSize = sizeof(DType)) {
			this->resizeBuffer(dimensions, elementSize);
		}

		Tensor(const Shape<IndexType> &shape, size_t elementSize = sizeof(DType)) {
			this->resizeBuffer(static_cast<const std::vector<IndexType> &>(shape), elementSize);
		}

		Tensor(uint8_t *buffer, size_t size, const std::vector<IndexType> &dimensions) {
			this->buffer = buffer;
			this->shape = dimensions;
			this->NrElements = this->getShape().getNrElements();
			this->ownAllocation = false;
		}

		Tensor(const Tensor &other) {
			this->resizeBuffer(other.getShape(), Ritsu::Tensor::DTypeSize);
			memcpy(this->buffer, other.buffer, other.getDatSize());
		}

		Tensor(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, nullptr);
			this->shape = other.shape;
			this->NrElements = other.NrElements;
			this->ownAllocation = other.ownAllocation;
		}

		~Tensor() {
			if (this->ownAllocation) {
				free(this->buffer);
			}
			this->buffer = nullptr;
		}

		auto &operator=(const Tensor &other) {
			this->resizeBuffer(other.getShape(), Ritsu::Tensor::DTypeSize);
			memcpy(this->buffer, other.buffer, other.getDatSize());

			return *this;
		}

		auto &operator=(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, nullptr);
			this->shape = other.shape;
			this->NrElements = other.NrElements;
			this->ownAllocation = other.ownAllocation;
			return *this;
		}

		// operations of data.
		template <typename U> inline const auto &getValue(const std::vector<IndexType> &location) const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeIndex(location);

			const U *addr = reinterpret_cast<const U *>(&buffer[index * DTypeSize]);
			return *addr;
		}

		template <typename U> inline auto &getValue(const std::vector<IndexType> &location) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeIndex(location);

			U *addr = reinterpret_cast<U *>(&buffer[index * DTypeSize]);
			return *addr;
		}

		template <typename U> inline auto &getValue(size_t index) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			U *addr = reinterpret_cast<U *>(&buffer[index * DTypeSize]);
			return *addr;
		}

		template <typename U> inline const auto &getValue(size_t index) const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const U *addr = reinterpret_cast<const U *>(&buffer[index * DTypeSize]);
			return *addr;
		}

		auto &operator+(const Tensor &tensor) {
			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<DType>(n) = this->getValue<DType>(n) + tensor.getValue<DType>(n);
			}
			return *this;
		}

		template <typename T> auto &operator+(const T &vec) {
			size_t nrElements = this->getNrElements();
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + vec;
			}
			return *this;
		}

		template <typename U> auto &operator-(const Tensor &tensor) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<float>(n) = this->getValue<float>(n) - tensor.getValue<float>(n);
			}
			return *this;
		}

		template <typename U> auto &operator*(const Tensor &tensor) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<DType>(n) = this->getValue<DType>(n) * tensor.getValue<DType>(n);
			}
			return *this;
		}

		auto &operator*(const DType &vec) {
			size_t nrElements = this->getNrElements();
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<float>(index) * vec;
			}
			return *this;
		}

		template <typename U> auto &operator/(const Tensor &tensor) {
			size_t nrElements = getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t n = 0; n < nrElements; n++) {
				this->getValue<DType>(n) = this->getValue<float>(n) / tensor.getValue<float>(n);
			}
			return *this;
		}

		void assignInitValue(const DType &initValue) {
			for (size_t i = 0; i < this->getNrElements(); i++) {
				this->getValue<DType>(i) = initValue;
			}
		}

		template <typename U> auto getSubset(size_t start, size_t end) const {
			Tensor subset = U(static_cast<uint8_t *>(&this->buffer[start * DTypeSize]), end - start, this->getShape());
			subset.ownAllocation = false;
			return subset;
		}

		template <typename U>
		auto getSubset(const std::vector<IndexType> &start, const std::vector<IndexType> &end) const {
			Tensor
				subset; // = T(static_cast<uint8_t *>(&this->buffer[start * DTypeSize]), end - start, this->dimensions);
			subset.ownAllocation = false;
			return subset;
		}

		Tensor &flatten() {
			/*	flatten dim.	*/
			this->shape = {(IndexType)this->getNrElements()};
			return *this;
		}

		template <typename T> Tensor &append(T tensor) {
			/*	Resize.	*/
			return *this;
		}
		Tensor &append(Tensor &tensor) {
			/*	Resize.	*/
			return *this;
		}

		// TODO add std::cout istream

		DType operator[](const std::vector<IndexType> &location) const { return getValue<DType>(location); }
		DType &operator[](const std::vector<IndexType> &location) { return getValue<DType>(location); }

		void resizeBuffer(const Shape<IndexType> &shape, const size_t elementSize) {
			size_t total_elements = this->compute_number_elements(shape);

			if (this->buffer != nullptr && this->ownAllocation) {
			}

			const size_t nrBytesAllocate = total_elements * elementSize;

			this->buffer = static_cast<uint8_t *>(realloc(this->buffer, nrBytesAllocate));
			this->shape = shape;
			this->NrElements = this->compute_number_elements(this->getShape());
		}

		friend std::ostream &operator<<(std::ostream &os, Tensor &tensor) {
			size_t number_elements = tensor.getNrElements();

			for (int i = 0; i < number_elements; i++) {
				size_t index = i;
				DType value = tensor.getValue<DType>(i);
				os << value << ",";
			}

			return os;
		}

		const Shape<IndexType> &getShape() const { return this->shape; }

		inline size_t computeIndex(const std::vector<IndexType> &dim) const {
			size_t totalSize = 1;

			for (size_t i = dim.size() - 1; i >= dim.size(); i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize;
		}

		template <typename U> inline const U *getRawData() const { return reinterpret_cast<U *>(this->buffer); }

		inline size_t getNrElements() const { return this->NrElements; }
		inline size_t getDatSize() const { return this->getNrElements() * DTypeSize; }

		static bool verifyShape(const Tensor &tensorA, const Tensor &tensorB) {
			/*	*/
			if (tensorA.getShape() != tensorB.getShape()) {
				return false;
			}
			return true;
		}

		void Reshape(const Shape<IndexType> &newShape) { this->shape.Reshape(newShape); }

	  protected:
		inline size_t compute_number_elements(const std::vector<IndexType> &dims) const {
			size_t totalSize = 1;
#pragma omp for simd
			for (size_t i = 0; i < dims.size(); i++) {
				totalSize *= dims[i];
			}
			return totalSize;
		}

	  private:
		using TensorBuffer = union _buffer_t {
			uint8_t *buffer = nullptr;
			float *fbuffer;
		};

		size_t NrElements = 0;
		Shape<IndexType> shape;
		// TODO custom buffer for improved performance.
		// TODO  add shared data
		union {
			uint8_t *buffer = nullptr;
			float *fbuffer;
		};
		// TODO  add shared data
		bool ownAllocation = true;
	};
} // namespace Ritsu