#pragma once
#include "core/Shape.h"
#include <algorithm>
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

	  public:
		using DType = float;
		static constexpr size_t DTypeSize = sizeof(DType);
		using IndexType = unsigned int;
		static constexpr size_t IndexTypeSize = sizeof(IndexType);

		static_assert(std::is_floating_point<DType>::value || std::is_integral<DType>::value,
					  "Must be a decimal type(float/double/half) or integer.");

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
			/*	*/
			this->resizeBuffer(other.getShape(), Ritsu::Tensor::DTypeSize);

			/*	*/
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

			const U *addr = reinterpret_cast<const U *>(&this->buffer[index * DTypeSize]);
			return *addr;
		}

		template <typename U> inline auto &getValue(const std::vector<IndexType> &location) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeIndex(location);

			U *addr = reinterpret_cast<U *>(&this->buffer[index * DTypeSize]);
			return *addr;
		}

		template <typename U> inline auto &getValue(size_t index) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			U *addr = reinterpret_cast<U *>(&this->buffer[index * DTypeSize]);
			return *addr;
		}

		template <typename U> inline const auto &getValue(size_t index) const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const U *addr = reinterpret_cast<const U *>(&this->buffer[index * DTypeSize]);
			return *addr;
		}

		auto &operator+(const Tensor &tensor) {
			if (verifyShape(*this, tensor)) {
			}

			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + tensor.getValue<DType>(index);
			}
			return *this;
		}

		template <typename T> auto &operator+(const T &vec) {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			size_t nrElements = this->getNrElements();
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + vec;
			}
			return *this;
		}

		auto &operator-() {
			size_t nrElements = this->getNrElements();
			// #pragma omp parallel shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = -this->getValue<DType>(index);
			}
			return *this;
		}

		auto operator-() const {
			Tensor output(getShape());
			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(output)
			for (size_t index = 0; index < nrElements; index++) {
				output.getValue<DType>(index) = -this->getValue<DType>(index);
			}
			return output;
		}

		auto &operator-(const Tensor &tensor) {
			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) - tensor.getValue<DType>(index);
			}
			return *this;
		}

		auto &operator*(const Tensor &tensor) {
			size_t nrElements = this->getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) * tensor.getValue<DType>(index);
			}
			return *this;
		}

		friend auto operator*(const Tensor &tensorA, const Tensor &tensorB) {
			Tensor output(tensorA.getShape());
			size_t nrElements = tensorA.getNrElements();

#pragma omp parallel shared(tensorA, tensorB, output)
			for (size_t index = 0; index < nrElements; index++) {
				output.getValue<DType>(index) = tensorA.getValue<DType>(index) * tensorB.getValue<DType>(index);
			}
			return output;
		}

		template <typename U> Tensor &operator*(U vec) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Type Must Support addition operation.");

			size_t nrElements = this->getNrElements();
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) * vec;
			}
			return *this;
		}

		template <typename U> auto &operator/(const Tensor &tensor) {
			size_t nrElements = this->getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) / tensor.getValue<DType>(index);
			}
			return *this;
		}

		void assignInitValue(const DType &initValue) {
			for (size_t i = 0; i < this->getNrElements(); i++) {
				this->getValue<DType>(i) = initValue;
			}
		}

		template <typename U> auto getSubset(size_t start, size_t end, const Shape<IndexType> &newShape = {}) const {
			// TODO update shape
			if (newShape.getNrDimensions() == 0) {
			}
			Tensor subset = U(static_cast<uint8_t *>(&this->buffer[start * DTypeSize]), end - start, newShape);
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

		template <typename T> Tensor &append(const T tensor) {
			/*	Resize.	*/
			Shape<IndexType> newShape({1});

			/*	Add additional data.	*/
			this->resizeBuffer(newShape, DTypeSize);

			/*	Copy new Data.	*/

			return *this;
		}

		template <typename T> Tensor &insert(const T tensor) { return *this; }

		Tensor &dot(const Tensor &tensor) { return *this; }
		friend Tensor dot(const Tensor &tensorA, const Tensor &tensorB) { return tensorA; }

		Tensor &append(const Tensor &tensor) {
			/*	Resize.	*/
			Shape<IndexType> newShape = tensor.getShape() + this->getShape();
			const size_t address_offset = this->getNrElements();

			this->resizeBuffer(newShape, DTypeSize);

			/*	Add additional data.	*/

			return *this;
		}

		template <typename U> Tensor &cast() {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			const size_t cast_element_size = sizeof(U);

			/*	Resize.	*/
			if (this->element_size != cast_element_size) {
			}

			// Convert value.
			return *this;
		}

		Tensor &transpose() { return *this; }

		DType operator[](const std::vector<IndexType> &location) const { return this->getValue<DType>(location); }
		DType &operator[](const std::vector<IndexType> &location) { return this->getValue<DType>(location); }

		void resizeBuffer(const Shape<IndexType> &shape, const size_t elementSize) {
			const size_t total_elements = shape.getNrElements();

			if (this->buffer != nullptr && this->ownAllocation) {
				/*	*/
			}

			const size_t nrBytesAllocate = total_elements * elementSize;

			// TODO handle if not the same pointer is returned.
			this->buffer = static_cast<uint8_t *>(realloc(this->buffer, nrBytesAllocate));
			this->shape = shape;
			this->NrElements = total_elements;
		}

		friend std::ostream &operator<<(std::ostream &stream, Tensor &tensor) {
			size_t number_elements = tensor.getNrElements();

			for (int i = 0; i < number_elements; i++) {
				size_t index = i;
				DType value = tensor.getValue<DType>(i);
				stream << value << ",";
			}

			return stream;
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
		template <typename U> inline U *getRawData() { return reinterpret_cast<U *>(this->buffer); }

		inline size_t getNrElements() const { return this->NrElements; }
		inline size_t getDatSize() const { return this->getNrElements() * DTypeSize; }

		static bool verifyShape(const Tensor &tensorA, const Tensor &tensorB) noexcept {
			/*	*/
			return tensorA.getShape() == tensorB.getShape();
		}
		// TODO add template to allow multiple of primtive types.
		void reshape(const Shape<IndexType> &newShape) { this->shape.reshape(newShape); }

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
		uint32_t element_size;

	  public: // TOOD relocate
		static Tensor log10(const Tensor &tensorA) {
			Tensor output(tensorA.getShape());
#pragma omp parallel shared(tensorA, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<DType>(i) = static_cast<DType>(std::log10(tensorA.getValue<DType>(i)));
			}
			return output;
		}

		static Tensor abs(const Tensor &tensorA) {
			Tensor output(tensorA.getShape());

#pragma omp parallel shared(tensorA, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<DType>(i) = static_cast<DType>(std::abs(tensorA.getValue<DType>(i)));
			}
			return output;
		}

		template <typename U> static U mean(const Tensor &tensorA) {
			// TODO fix tup
			if (tensorA.getNrElements() == 0) {
				return 0;
			}
			return static_cast<U>(Math::mean<float>(tensorA.getRawData<float>(), tensorA.getNrElements()));
		}

		template <typename U> static U variance(const Tensor &tensorA, const U mean) {
			// TODO fix tup
			return static_cast<U>(Math::variance<float>(tensorA.getRawData<float>(), tensorA.getNrElements(), mean));
		}

		static Tensor zero(const Shape<IndexType> &shape) {
			Tensor zeroTesnor(shape);

			memset(zeroTesnor.getRawData<void *>(), 0, zeroTesnor.getNrElements() * DTypeSize);

			return zeroTesnor;
		}

		static Tensor OneShot(const Shape<IndexType> &shape, size_t value) {
			Tensor tensor = Tensor::zero(shape);
			tensor.getValue<float>(value) = 1.0f;

			return tensor;
		}
	};
} // namespace Ritsu