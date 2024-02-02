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
#include "core/Shape.h"
#include <algorithm>
#include <atomic>
#include <cassert>
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
#include <typeinfo>
#include <utility>
#include <vector>

namespace Ritsu {

	/**
	 * @brief Multi dimensional array
	 *
	 */
	// TODO: align
	template <typename T = float> class Tensor {
	  public:
		/*	*/
		using DType = T;
		static constexpr unsigned int DTypeSize = sizeof(T);
		/*	*/
		using IndexType = unsigned int;
		static constexpr unsigned int IndexTypeSize = sizeof(IndexType);

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
		// template <typename Param>
		Tensor(const std::vector<IndexType> &dimensions, const size_t elementSize = DTypeSize) {
			this->resizeBuffer(dimensions, elementSize);
			this->typeinfo = &typeid(DType);
		}

		Tensor(const Shape<IndexType> &shape, const size_t elementSize = DTypeSize) {
			this->resizeBuffer(static_cast<const std::vector<IndexType> &>(shape), elementSize);
			this->typeinfo = &typeid(DType);
		}

		Tensor(uint8_t *buffer, const size_t sizeInBytes, const std::vector<IndexType> &dimensions,
			   const size_t elementSize = DTypeSize) {
			/*	TODO: verify */
			this->memoryBuffer.buffer.buffer = buffer;
			this->memoryBuffer.ownerUid = reinterpret_cast<size_t>(buffer);
			this->memoryBuffer.uid = reinterpret_cast<size_t>(this);
			this->memoryBuffer.allocationSize = sizeInBytes;
			this->shape = dimensions;
			this->NrElements = this->getShape().getNrElements();
			this->memoryBuffer.element_size = elementSize;
			this->typeinfo = &typeid(DType);
		}

		Tensor(const Tensor &other) {
			this->resizeBuffer(other.getShape(), other.memoryBuffer.element_size);
			std::memcpy(this->memoryBuffer.buffer.buffer, other.memoryBuffer.buffer.buffer, other.getDatSize());
			this->typeinfo = other.typeinfo;
		}

		Tensor(Tensor &&other) {
			this->memoryBuffer.buffer.buffer = std::exchange(other.memoryBuffer.buffer.buffer, nullptr);
			this->shape = std::move(other.shape);
			this->NrElements = other.NrElements;
			this->memoryBuffer.allocationSize = other.memoryBuffer.allocationSize;
			this->memoryBuffer.nrReferences.store(other.memoryBuffer.nrReferences.load());
			this->memoryBuffer.ownerUid = other.memoryBuffer.ownerUid;
			this->memoryBuffer.uid = other.memoryBuffer.uid;
			this->memoryBuffer.element_size = other.memoryBuffer.element_size;
			this->typeinfo = other.typeinfo;
		}

		~Tensor() {

			/*	*/
			if (this->memoryBuffer.nrReferences.fetch_sub(1) == 0 && this->memoryBuffer.buffer.buffer != nullptr &&
				this->ownAllocation()) {

				free(this->memoryBuffer.buffer.buffer);
				this->memoryBuffer.buffer.buffer = nullptr;
				this->memoryBuffer.allocationSize = 0;
				this->memoryBuffer.ownerUid = 0;
				this->memoryBuffer.uid = 0;
			}
		}

		auto &operator=(const Tensor &other) {
			this->resizeBuffer(other.getShape(), other.memoryBuffer.element_size);
			std::memcpy(this->memoryBuffer.buffer.buffer, other.memoryBuffer.buffer.buffer, other.getDatSize());
			this->typeinfo = other.typeinfo;
			return *this;
		}

		auto &operator=(Tensor &&other) {
			this->memoryBuffer.buffer.buffer = std::exchange(other.memoryBuffer.buffer.buffer, nullptr);
			this->shape = std::move(other.shape);
			this->NrElements = other.NrElements;
			this->memoryBuffer.allocationSize = other.memoryBuffer.allocationSize;
			this->memoryBuffer.nrReferences.store(other.memoryBuffer.nrReferences.load());
			this->memoryBuffer.element_size = other.memoryBuffer.element_size;

			// TODO: fix own.
			this->memoryBuffer.ownerUid = other.memoryBuffer.ownerUid;
			this->memoryBuffer.uid = other.memoryBuffer.uid;

			this->typeinfo = other.typeinfo;
			return *this;
		}

		bool operator==(const Tensor &tensor) const {

			/*	Same address => equal.	*/
			if (static_cast<const void *>(&tensor) == static_cast<const void *>(this)) {
				return true;
			}

			/*	Check internal buffer.	*/
			if (static_cast<const void *>(&tensor.memoryBuffer.buffer.buffer) ==
				static_cast<const void *>(this->memoryBuffer.buffer.buffer)) {
				return true;
			}

			/*	Check same shape.	*/
			if (this->getNrElements() != tensor.getNrElements() || this->getDatSize() != tensor.getDatSize()) {
				return false;
			}

			if (this->getShape() != tensor.getShape()) {
				return false;
			}

			// Last check, see if the content matches.
			if (memcmp(this->memoryBuffer.buffer.buffer, tensor.memoryBuffer.buffer.buffer, this->getDatSize()) != 0) {
				return false;
			}

			return true;
		}

		bool operator!=(const Tensor &tensor) const { return !(*this == tensor); }

		// Dtype
		const std::type_info &getDType() const noexcept { return *this->typeinfo; }
		const std::type_info *typeinfo;

		// operations of data.
		template <typename U = DType> inline U getValue(const std::vector<IndexType> &location) const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeShape2Index(location);
			return Tensor::getValue<U>((IndexType)index);
		}

		template <typename U = DType> inline U &getValue(const std::vector<IndexType> &location) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t index = this->computeShape2Index(location);
			return Tensor::getValue<U>((IndexType)index);
		}

		template <typename U = DType> inline U &getValue(const IndexType index) noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			U *addr = reinterpret_cast<U *>(&this->memoryBuffer.buffer.buffer[index * this->memoryBuffer.element_size]);
			return *addr;
		}

		template <typename U = DType> inline U getValue(const IndexType index) const noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			const U *addr =
				reinterpret_cast<const U *>(&this->memoryBuffer.buffer.buffer[index * this->memoryBuffer.element_size]);
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

		template <typename U> auto &operator+(const U &vec) {
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

		auto &operator%(const Tensor &tensor) noexcept {
			*this = matrixMultiply(*this, tensor);
			return *this;
		}

		friend auto operator%(const Tensor &tensorA, const Tensor &tensorB) { return matrixMultiply(tensorA, tensorB); }

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

			const size_t nrElements = this->getNrElements();

#pragma omp parallel for simd
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) * vec;
			}

			return *this;
		}

		template <typename U> Tensor operator*(U vec) const noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Type Must Support addition operation.");
			Tensor tmp = *this;

			const size_t nrElements = this->getNrElements();

#pragma omp parallel for simd
			for (size_t index = 0; index < nrElements; index++) {
				tmp.getValue<DType>(index) = this->getValue<DType>(index) * vec;
			}

			return tmp;
		}

		template <typename U> auto &operator/(const Tensor &tensor) {
			size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) / tensor.getValue<DType>(index);
			}
			return *this;
		}

		template <typename U> void assignInitValue(const U initValue) noexcept {
			const IndexType nrElements = this->getNrElements();

#pragma omp parallel for simd shared(nrElements, initValue)
			for (size_t i = 0; i < nrElements; i++) {
				this->getValue<DType>(i) = static_cast<DType>(initValue);
			}
		}

		Tensor getSubset(size_t start, size_t end, const Shape<IndexType> &newShape = {}) const {
			// TODO update shape
			if (newShape.getNrDimensions() == 0) {
			}
			//	this->memoryBuffer.nrReferences.fetch_add(1);
			Tensor subset = Tensor(static_cast<uint8_t *>(&this->memoryBuffer.buffer.buffer[start * DTypeSize]),
								   end - start, newShape, this->memoryBuffer.element_size);

			return subset;
		}

		Tensor &flatten() noexcept {
			/*	flatten dim.	*/
			this->shape = {(IndexType)this->getNrElements()};
			return *this;
		}

		Tensor &reduce() noexcept {
			/*	flatten dim.	*/
			this->shape.reduce();
			return *this;
		}

		template <typename U> Tensor &insert(const U tensor) { return *this; }

		DType dot(const Tensor &tensor) const noexcept { return Tensor::dot(*this, tensor); }

		static DType dot(const Tensor &tensorA, const Tensor &tensorB) noexcept {
			// TODO: determine type.
			return Math::dot<DType>(tensorA.getRawData<DType>(), tensorB.getRawData<DType>(), tensorA.getNrElements());
		}

		Tensor &pow(const DType value) noexcept {
			Math::pow(value, this->getRawData<DType>(), this->getNrElements());
			return *this;
		}

		Tensor mean(const Tensor &tensorA) noexcept { return fromArray({mean<DType>(tensorA)}); }

		Tensor &append(const Tensor &tensor) { // TODO: add axis

			/*	Resize.	*/
			const Shape<IndexType> newShape = tensor.getShape() + this->getShape();
			const size_t original_address_offset = this->getDatSize();

			/*	*/
			const size_t element_size = this->memoryBuffer.element_size;
			bool cast = false;
			if (tensor.memoryBuffer.element_size != element_size) {
				// TODO: cast
				cast = true;
			}

			/*	*/
			this->resizeBuffer(newShape, element_size);

			/*	Add additional data.	*/
			if (!cast) {
				std::memcpy(&this->getRawData<uint8_t *>()[original_address_offset],
							tensor.getRawData<const uint8_t *>(), tensor.getDatSize());
			} else {
				// TODO: impl cast
			}

			return *this;
		}

		// TODO: add init_list
		template <typename U> Tensor &append(const U value) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			/*	Resize.	*/
			this->shape = this->shape + Shape<IndexType>({1});

			/*	Add additional data.	*/
			this->resizeBuffer(this->shape, DTypeSize);

			/*	Copy new Data.	*/
			this->getValue<DType>(this->getNrElements() - 1) = value;

			return *this;
		}

		template <typename U> Tensor<U> &cast() {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			const size_t cast_element_size = sizeof(U);

			/*	Resize.	*/
			if (this->memoryBuffer.element_size != cast_element_size) {

				Tensor<U> tmp = Tensor<U>(getShape(), sizeof(U));

#pragma omp parallel for simd
				for (size_t i = 0; i < this->getNrElements(); i++) {
					tmp.template getValue<U>(i) = static_cast<DType>(this->getValue<DType>(i));
				}

				Tensor<U> &ref = reinterpret_cast<Tensor<U> &>(*this);
				ref = std::move(tmp);
				return ref;
			}

			this->memoryBuffer.element_size = cast_element_size;
#pragma omp parallel for simd
			for (size_t i = 0; i < this->getNrElements(); i++) {
				this->getValue<U>(i) = static_cast<DType>(this->getValue<DType>(i));
			}

			Tensor<U> &ref = reinterpret_cast<Tensor<U> &>(*this);
			// Convert value.
			return ref;
		}

		template <typename U> Tensor cast() const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			const size_t cast_element_size = sizeof(U);

			/*	Resize.	*/
			if (this->memoryBuffer.element_size != cast_element_size) {
			}
			Tensor tensor = Tensor(this->getShape(), sizeof(U));
#pragma omp parallel for simd
			for (size_t i = 0; i < this->getNrElements(); i++) {
				tensor.getValue<U>(i) = static_cast<DType>(this->getValue<DType>(i));
			}
			// Convert value.
			return *this;
		}

		Tensor &transpose() noexcept {
			this->shape.transpose();
			return *this;
		}

		DType operator[](const std::vector<IndexType> &location) const { return this->getValue<DType>(location); }
		DType &operator[](const std::vector<IndexType> &location) { return this->getValue<DType>(location); }

		void resizeBuffer(const Shape<IndexType> &shape, const size_t elementSize) {
			const size_t total_nr_elements = shape.getNrElements();

			if (this->memoryBuffer.buffer.buffer != nullptr && !this->ownAllocation()) {
				/*	*/
				throw std::runtime_error("Can not allocate on not owned tensor.");
			}

			if (total_nr_elements <= 0) {
				throw std::runtime_error("Must be greater than 0");
			}

			/*	Compute size in bytes, aligned.	*/
			const size_t nrByteUnAligned = total_nr_elements * elementSize;
			const size_t nrBytesAllocateAligned = Math::align<size_t>(nrByteUnAligned, 4);

			// TODO handle if not the same pointer is returned.

			/*	Set ownership if never allocated before.	*/
			if (this->memoryBuffer.buffer.dbuffer == nullptr) {
				this->memoryBuffer.uid = reinterpret_cast<size_t>(this);
				if (this->memoryBuffer.ownerUid == 0) {
					this->memoryBuffer.ownerUid = this->memoryBuffer.uid;
				}
			}

			/*	*/
			const void *prevBuf = this->memoryBuffer.buffer.buffer;
			this->memoryBuffer.buffer.buffer =
				static_cast<uint8_t *>(realloc(this->memoryBuffer.buffer.buffer, nrBytesAllocateAligned));

			this->memoryBuffer.allocationSize = nrBytesAllocateAligned;

			if (this->memoryBuffer.buffer.buffer == nullptr) {
				throw std::runtime_error("Error");
			}
			if (this->memoryBuffer.buffer.buffer != prevBuf && prevBuf != nullptr) {
				/*	*/
				throw std::runtime_error("Error");
			}

			this->shape = shape;
			this->NrElements = total_nr_elements;
			this->memoryBuffer.element_size = elementSize;
		}

		friend std::ostream &operator<<(std::ostream &stream, Tensor &tensor) {
			const IndexType number_elements = tensor.getNrElements();

			/*	*/
			for (IndexType index = 0; index < number_elements; index++) {
				DType value = tensor.getValue<DType>(index);
				stream << value << ",";
			}

			return stream;
		}

		const Shape<IndexType> &getShape() const { return this->shape; }

		inline size_t computeShape2Index(const std::vector<IndexType> &dim) const noexcept {
			return Shape<IndexType>::computeIndex(dim);
		}

		template <typename U> inline constexpr const U *getRawData() const noexcept {
			return reinterpret_cast<const U *>(this->memoryBuffer.buffer.buffer);
		}
		template <typename U> inline constexpr U *getRawData() noexcept {
			return reinterpret_cast<U *>(this->memoryBuffer.buffer.buffer);
		}

		inline IndexType getNrElements() const noexcept { return this->NrElements; }
		inline IndexType getDatSize() const noexcept { return this->getNrElements() * this->memoryBuffer.element_size; }
		inline IndexType getInternalDatSize() const noexcept { return this->memoryBuffer.allocationSize; }
		inline size_t getElementSize() const noexcept { return this->memoryBuffer.element_size; }

		static bool verifyShape(const Tensor &tensorA, const Tensor &tensorB) noexcept {
			/*	*/
			return tensorA.getShape() == tensorB.getShape();
		}

		// TODO add template to allow multiple of primtive types.
		Tensor &reshape(const Shape<IndexType> &newShape) {
			this->shape.reshape(newShape);
			return *this;
		}

	  private:
		using TensorBuffer = union _buffer_t {
			uint8_t *buffer = nullptr;
			float *fbuffer;
			DType *dbuffer;
		};
		using InternalBuffer = struct internal_buffer_t {
			TensorBuffer buffer;
			size_t allocationSize;
			size_t uid = 0;
			std::atomic_int32_t nrReferences = {1};
			size_t ownerUid = 0;
			uint32_t element_size = 0;
		};

		size_t NrElements = 0;		 /*	Cache value of shape number of elements.*/
		Shape<IndexType> shape;		 /*	*/
		InternalBuffer memoryBuffer; /*	*/

		inline bool ownAllocation() const noexcept { return this->memoryBuffer.uid == this->memoryBuffer.ownerUid; }

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
				//	output.getValue<Tensor::DType>(i) = static_cast<Tensor::DType>(std::abs(
				// tensorA.getValue<Tensor::DType>(i)));
			}
			return output;
		}

		template <typename U> static U mean(const Tensor &tensorA) noexcept {

			if (tensorA.getNrElements() == 0) {
				return 0;
			}
			return static_cast<U>(Math::mean<DType>(tensorA.getRawData<DType>(), tensorA.getNrElements()));
		}

		template <typename U> static Tensor mean(const Tensor &tensorA, int axis) noexcept {

			if (tensorA.getNrElements() == 0) {
				return {};
			}

			Tensor result({tensorA.getShape().getAxisDimensions(axis)});

			/*	*/
			for (size_t i = 0; i < tensorA.getShape().getAxisDimensions(axis); i++) {

				const DType *data = tensorA.getRawData<DType>();

				size_t elements = tensorA.getShape().getSubShape(0).getNrElements();

				result.getValue(i) =
					static_cast<U>(Math::mean<DType>(data, tensorA.getShape().getAxisDimensions(1))); // TODO:
			}

			return result;
		}

		template <typename U> static U variance(const Tensor &tensorA, const U mean) noexcept {

			return static_cast<U>(Math::variance<DType>(tensorA.getRawData<DType>(), tensorA.getNrElements(), mean));
		}

		template <typename U> static Tensor variance(const Tensor &tensorA, const U mean, int axis) noexcept {
			return tensorA;
		}

		static Tensor zero(const Shape<IndexType> &shape) {
			Tensor zeroTesnor(shape);

			/*	Zero out memory.	*/
			memset(zeroTesnor.getRawData<void *>(), 0, zeroTesnor.getDatSize());

			return zeroTesnor;
		}

		static Tensor oneShot(const Shape<IndexType> &shape, size_t value) {
			Tensor tensor = Tensor::zero(shape); // TODO: improved performance.
			tensor.getValue<float>(value) = 1.0f;

			return tensor;
		}

		static Tensor matrixMultiply(const Tensor &tensorA, const Tensor &tensorB) {

			// TODO verify shape.

			/*	TODO:*/
			Tensor output(Shape<IndexType>({tensorA.getShape()[0], tensorB.getShape()[1]}));

			Tensor::matrixMultiply(tensorA, tensorB, output);

			return output;
		}

		static Tensor matrixMultiply(const Tensor &tensorA, const Tensor &tensorB, Tensor &output) {

			// TODO verify shape.

#pragma omp parallel for
			for (size_t y = 0; y < tensorA.getShape()[0]; y++) {
				DType sum = 0;

#pragma omp simd reduction(+ : sum)
				for (size_t x = 0; x < tensorA.getShape()[1]; x++) {

					const size_t index = y * tensorA.getShape()[0] + x;

					sum += tensorA.getValue<DType>(index) * tensorB.getValue<DType>(y);
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

		template <typename U> static Tensor fromArray(const std::initializer_list<U> &list) {

			Tensor tensor({(IndexType)list.size()}, sizeof(U));

			IndexType index = 0;
			for (typename std::initializer_list<U>::const_iterator i = list.begin(); i != list.end(); i++) {
				const U value = *i;

				tensor.getValue<DType>(index++) = static_cast<DType>(value);
			}

			return tensor;
		}

		template <typename U> static Tensor split(Tensor &list) { return Tensor({1}, sizeof(U)); }
	};
} // namespace Ritsu