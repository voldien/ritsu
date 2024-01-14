#pragma once
#include "core/Shape.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <istream>
#include <memory>
#include <omp.h>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Ritsu {

	/**
	 * @brief Multi dimensional array
	 *
	 */
	// TODO: align
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
		Tensor(const std::vector<IndexType> &dimensions, const size_t elementSize = DTypeSize) {
			this->ownAllocation = true;
			this->resizeBuffer(dimensions, elementSize);
		}

		Tensor(const Shape<IndexType> &shape, const size_t elementSize = DTypeSize) {
			this->ownAllocation = true;
			this->resizeBuffer(static_cast<const std::vector<IndexType> &>(shape), elementSize);
		}

		Tensor(uint8_t *buffer, const size_t sizeInBytes, const std::vector<IndexType> &dimensions,
			   const size_t elementSize = DTypeSize) {
			this->buffer = buffer;
			this->shape = dimensions;
			this->NrElements = this->getShape().getNrElements();
			this->element_size = elementSize;
			this->ownAllocation = false;
		}

		Tensor(const Tensor &other) {
			this->ownAllocation = true;
			this->resizeBuffer(other.getShape(), Ritsu::Tensor::DTypeSize);
			memcpy(this->buffer, other.buffer, other.getDatSize());
		}

		Tensor(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, nullptr);
			this->shape = std::move(other.shape);
			this->NrElements = other.NrElements;
			this->ownAllocation = other.ownAllocation;
			this->element_size = other.element_size;
		}

		~Tensor() {
			if (this->ownAllocation) {
				free(this->buffer);
			}
			this->buffer = nullptr;
		}

		auto &operator=(const Tensor &other) {
			/*	*/
			this->ownAllocation = true;
			this->resizeBuffer(other.getShape(), Ritsu::Tensor::DTypeSize);
			/*	*/
			memcpy(this->buffer, other.buffer, other.getDatSize());

			return *this;
		}

		auto &operator=(Tensor &&other) {
			this->buffer = std::exchange(other.buffer, nullptr);
			this->shape = std::move(other.shape);
			this->NrElements = other.NrElements;
			this->element_size = other.element_size;
			this->ownAllocation = other.ownAllocation;
			return *this;
		}

		bool operator==(const Tensor &tensor) const {

			/*	Same address => equal.	*/
			if (static_cast<const void *>(&tensor) == static_cast<const void *>(this)) {
				return true;
			}

			if (static_cast<const void *>(&tensor.buffer) == static_cast<const void *>(this->buffer)) {
				return true;
			}

			// TODO: add shape.
			if (this->getNrElements() != tensor.getNrElements() || this->getDatSize() != tensor.getDatSize()) {
				return false;
			}

			// TODO: validate.
			if (memcmp(this->buffer, tensor.buffer, this->getDatSize()) != 0) {
				return false;
			}

			return true;
		}

		bool operator!=(const Tensor &tensor) const { return !(*this == shape); }

		// operations of data.
		template <typename U> inline U getValue(const std::vector<IndexType> &location) const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeIndex(location);
			return Tensor::getValue<U>((IndexType)index);
		}

		template <typename U> inline U &getValue(const std::vector<IndexType> &location) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeIndex(location);
			return Tensor::getValue<U>((IndexType)index);
		}

		template <typename U> inline U &getValue(const IndexType index) noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			U *addr = reinterpret_cast<U *>(&this->buffer[index * this->element_size]);
			return *addr;
		}

		template <typename U> inline U getValue(const IndexType index) const noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			const U *addr = reinterpret_cast<const U *>(&this->buffer[index * this->element_size]);
			return *addr;
		}

		auto &operator+(const Tensor &tensor) {
			if (verifyShape(*this, tensor)) {
			}

			const size_t nrElements = this->getNrElements();
#pragma omp parallel for simd shared(tensor, nrElements)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + tensor.getValue<DType>(index);
			}
			return *this;
		}

		template <typename T> auto &operator+(const T &vec) {
			static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t nrElements = this->getNrElements();
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + vec;
			}
			return *this;
		}

		auto &operator-() noexcept {
			size_t nrElements = this->getNrElements();
			// #pragma omp parallel shared(tensor)
			#pragma omp parallel for simd
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = -this->getValue<DType>(index);
			}
			return *this;
		}

		auto operator-() const {
			Tensor output(getShape());
			const size_t nrElements = this->getNrElements();
#pragma omp parallel for shared(output)
			for (size_t index = 0; index < nrElements; index++) {
				output.getValue<DType>(index) = -this->getValue<DType>(index);
			}
			return output;
		}

		auto &operator-(const Tensor &tensor) noexcept {
			size_t nrElements = this->getNrElements();
#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) - tensor.getValue<DType>(index);
			}
			return *this;
		}

		Tensor operator-(const Tensor &tensor) const {
			Tensor tmp = *this;
			size_t nrElements = this->getNrElements();
#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				tmp.getValue<DType>(index) = this->getValue<DType>(index) - tensor.getValue<DType>(index);
			}
			return tmp;
		}

		auto &operator*(const Tensor &tensor) noexcept {
			size_t nrElements = this->getNrElements();
#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) * tensor.getValue<DType>(index);
			}
			return *this;
		}

		friend auto operator*(const Tensor &tensorA, const Tensor &tensorB) {

			// TODO: verify shape

			Tensor output(tensorA.getShape());
			size_t nrElements = tensorA.getNrElements();

#pragma omp parallel for shared(tensorA, tensorB, output)
			for (size_t index = 0; index < nrElements; index++) {
				output.getValue<DType>(index) = tensorA.getValue<DType>(index) * tensorB.getValue<DType>(index);
			}
			return output;
		}

		template <typename U> Tensor &operator*(U vec) noexcept {
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

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) / tensor.getValue<DType>(index);
			}
			return *this;
		}

		void assignInitValue(const DType initValue) noexcept {
			const IndexType nrElements = this->getNrElements();

#pragma omp parallel for simd
			for (size_t i = 0; i < nrElements; i++) {
				this->getValue<DType>(i) = initValue;
			}
		}

		Tensor getSubset(size_t start, size_t end, const Shape<IndexType> &newShape = {}) const {
			// TODO update shape
			if (newShape.getNrDimensions() == 0) {
			}
			Tensor subset = Tensor(static_cast<uint8_t *>(&this->buffer[start * DTypeSize]), end - start, newShape,
								   this->element_size);
			return subset;
		}

		Tensor getSubset(const std::vector<IndexType> &start, const std::vector<IndexType> &end) const {
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

		DType dot(const Tensor &tensor) const noexcept { return Tensor::dot(*this, tensor); }

		static DType dot(const Tensor &tensorA, const Tensor &tensorB) noexcept {
			return Math::dot<DType>(tensorA.getRawData<DType>(), tensorB.getRawData<DType>(), tensorA.getNrElements());
		}

		Tensor &pow(const DType value) noexcept {
			Math::pow<DType>(value, this->getRawData<DType>(), this->getNrElements());
			return *this;
		}

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

			this->element_size = cast_element_size;

			// Convert value.
			return *this;
		}

		template <typename U> Tensor cast() const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			const size_t cast_element_size = sizeof(U);

			/*	Resize.	*/
			if (this->element_size != cast_element_size) {
			}

			// this->element_size = cast_element_size;

			// Convert value.
			return *this;
		}

		Tensor &transpose() { return *this; }

		DType operator[](const std::vector<IndexType> &location) const { return this->getValue<DType>(location); }
		DType &operator[](const std::vector<IndexType> &location) { return this->getValue<DType>(location); }

		void resizeBuffer(const Shape<IndexType> &shape, const size_t elementSize) {
			const size_t total_elements = shape.getNrElements();

			if (this->buffer != nullptr && !this->ownAllocation) {
				/*	*/
				throw std::runtime_error("Invalid State");
			}

			if (total_elements == 0) {
				throw std::runtime_error("Must be greater than 0");
			}

			const size_t nrBytesAllocate = Math::align<size_t>(total_elements * elementSize, 4);

			// TODO handle if not the same pointer is returned.
			this->buffer = static_cast<uint8_t *>(realloc(this->buffer, nrBytesAllocate));
			if (this->buffer == nullptr) {
				throw std::runtime_error("Error");
			}

			this->shape = shape;
			this->NrElements = total_elements;
			this->element_size = elementSize;
		}

		friend std::ostream &operator<<(std::ostream &stream, Tensor &tensor) {
			const size_t number_elements = tensor.getNrElements();

			for (int i = 0; i < number_elements; i++) {
				size_t index = i;
				DType value = tensor.getValue<DType>(i);
				stream << value << ",";
			}

			return stream;
		}

		const Shape<IndexType> &getShape() const { return this->shape; }

		inline size_t computeIndex(const std::vector<IndexType> &dim) const noexcept {
			size_t totalSize = 1;

			for (size_t i = dim.size() - 1; i >= dim.size(); i--) {
				totalSize *= dim[i];
			}
			totalSize += dim[0];
			return totalSize;
		}

		template <typename U> inline constexpr const U *getRawData() const noexcept {
			return reinterpret_cast<const U *>(this->buffer);
		}
		template <typename U> inline constexpr U *getRawData() noexcept { return reinterpret_cast<U *>(this->buffer); }

		inline IndexType getNrElements() const noexcept { return this->NrElements; }
		inline IndexType getDatSize() const noexcept { return this->getNrElements() * this->element_size; }

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
		uint32_t element_size = 0;

	  public: // TOOD relocate
		static Tensor log10(const Tensor &tensorA) {
			Tensor output(tensorA.getShape());
#pragma omp parallel for shared(tensorA, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<DType>(i) = static_cast<DType>(std::log10(tensorA.getValue<DType>(i)));
			}
			return output;
		}

		static Tensor abs(const Tensor &tensorA) {
			Tensor output(tensorA.getShape());

#pragma omp parallel for shared(tensorA, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<DType>(i) = static_cast<DType>(std::abs(tensorA.getValue<DType>(i)));
			}
			return output;
		}

		template <typename U> static U mean(const Tensor &tensorA) noexcept {

			if (tensorA.getNrElements() == 0) {
				return 0;
			}
			return static_cast<U>(Math::mean<float>(tensorA.getRawData<float>(), tensorA.getNrElements()));
		}

		template <typename U> static U variance(const Tensor &tensorA, const U mean) noexcept {

			return static_cast<U>(Math::variance<float>(tensorA.getRawData<float>(), tensorA.getNrElements(), mean));
		}

		static Tensor zero(const Shape<IndexType> &shape, const size_t elementSize = DTypeSize) {
			Tensor zeroTesnor(shape);

			memset(zeroTesnor.getRawData<void *>(), 0, zeroTesnor.getNrElements() * DTypeSize);

			return zeroTesnor;
		}

		static Tensor OneShot(const Shape<IndexType> &shape, size_t value) {
			Tensor tensor = Tensor::zero(shape); // TODO: improved performance.
			tensor.getValue<float>(value) = 1.0f;

			return tensor;
		}

		static Tensor matrixMultiply(const Tensor &tensorA, const Tensor &tensorB) {

			Tensor output(Shape<IndexType>({tensorA.getShape()[0], tensorB.getShape()[1]}));

#pragma omp parallel for simd
			for (size_t y = 0; y < tensorA.getShape()[0]; y++) {
				DType sum = 0;

				for (size_t x = 0; x < tensorA.getShape()[1]; x++) {

					size_t index = y * tensorA.getShape()[0] + x;
					sum = tensorA.getValue<DType>(index) * tensorB.getValue<DType>(y);
				}
				output.getValue<DType>(y) = sum;
			}

			return output;
		}

		static Tensor equal(const Tensor &tensorA, const Tensor &tensorB) {

			Tensor output(tensorA.getShape(), sizeof(uint8_t));

#pragma omp parallel for shared(tensorA, tensorB, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<DType>(i) = tensorA.getValue<DType>(i) == tensorB.getValue<DType>(i);
			}
			return output;
		}

		template <typename T> static Tensor fromArray(const std::initializer_list<T> &list) {
			return Tensor({1}, sizeof(T));
		}
	};
} // namespace Ritsu