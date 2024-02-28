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
#include "RitsuDef.h"
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
#include <limits>
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
	template <typename T = float, unsigned int alignment = 16> class Tensor {
	  public:
		/*	*/
		using DType = T;
		static constexpr unsigned int DTypeSize = sizeof(T);

		/*	*/
		using IndexType = unsigned int;
		static constexpr const unsigned int IndexTypeSize = sizeof(IndexType);
		static constexpr const unsigned int alignmentByte = alignment;

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
		Tensor(const std::vector<IndexType> &dimensions, const size_t elementSize = DTypeSize) {
			this->resizeBuffer(dimensions, elementSize);
			this->typeinfo = &typeid(DType);
		}

		Tensor(const Shape<IndexType> &shape, const size_t elementSize = DTypeSize) {
			this->resizeBuffer(static_cast<const std::vector<IndexType> &>(shape), elementSize);
			this->typeinfo = &typeid(DType);
		}

		Tensor(const uint8_t *buffer, const size_t sizeInBytes, const std::vector<IndexType> &dimensions,
			   const size_t elementSize = DTypeSize) {

			const size_t nrElementsInBuffer = sizeInBytes / elementSize;
			if (Shape<IndexType>::computeNrElements(dimensions) < nrElementsInBuffer) {
				throw InvalidIndexException("Invalid Shape.");
			}

			this->memoryBuffer.buffer.data = const_cast<uint8_t *>(buffer);
			this->memoryBuffer.ownerUid = reinterpret_cast<size_t>(buffer);
			this->memoryBuffer.uid = reinterpret_cast<size_t>(this);
			this->memoryBuffer.allocationSize = sizeInBytes;
			this->shape = dimensions;
			this->NrElements = this->getShape().getNrElements();
			this->memoryBuffer.element_size = elementSize;
			this->memoryBuffer.memoryShape = dimensions;
			this->typeinfo = &typeid(DType);
		}

		Tensor(const Shape<IndexType> &newShape, const Shape<IndexType> &offsetShape, const Tensor &parent) {

			/*	*/ // TODO: fix offset
			const size_t offsetMemory = offsetShape.getNrElements() * parent.getElementSize();
			this->memoryBuffer.buffer.data = &parent.memoryBuffer.buffer.data[offsetMemory];

			this->memoryBuffer.ownerUid = reinterpret_cast<size_t>(parent.memoryBuffer.ownerUid);
			this->memoryBuffer.uid = reinterpret_cast<size_t>(this);
			/*	*/
			this->memoryBuffer.allocationSize = newShape.getNrElements() * parent.getElementSize();
			this->memoryBuffer.memoryShape = parent.memoryBuffer.memoryShape;
			this->NrElements = newShape.getNrElements();
			this->memoryBuffer.element_size = parent.getElementSize();
			this->typeinfo = parent.typeinfo;

			/*	*/
			this->shape = newShape;
		}

		Tensor(const Tensor &other) {
			this->resizeBuffer(other.getShape(), other.memoryBuffer.element_size);

			/*	Transfer data.	*/
			const size_t dataSizeInBytes = other.getDatSize();
			std::memcpy(this->memoryBuffer.buffer.data, other.memoryBuffer.buffer.data, dataSizeInBytes);

			this->typeinfo = other.typeinfo;
		}

		Tensor(Tensor &&other) noexcept {
			this->release();

			/*	*/
			this->memoryBuffer.buffer.data = std::exchange(other.memoryBuffer.buffer.data, nullptr);
			this->shape = std::move(other.shape);

			/*	*/
			this->NrElements = other.NrElements;
			this->memoryBuffer.allocationSize = other.memoryBuffer.allocationSize;
			this->memoryBuffer.nrReferences.store(other.memoryBuffer.nrReferences.load());
			this->memoryBuffer.ownerUid = other.memoryBuffer.ownerUid;
			this->memoryBuffer.uid = other.memoryBuffer.uid;
			this->memoryBuffer.element_size = other.memoryBuffer.element_size;
			this->memoryBuffer.memoryShape = std::move(other.memoryBuffer.memoryShape);
			this->typeinfo = other.typeinfo;
		}

		~Tensor() noexcept {
			/*	*/
			this->release();
		}

		void release() noexcept {

			/*	*/
			this->memoryBuffer.nrReferences.fetch_sub(1);

			/*	Determine if the memory can be released.	*/
			if (this->memoryBuffer.nrReferences.load() == 0 && this->ownAllocation() &&
				this->memoryBuffer.buffer.data != nullptr) {

				free(this->memoryBuffer.buffer.data);
				this->memoryBuffer.buffer.data = nullptr;
			}
		}

		auto &operator=(const Tensor &other) {

			/*	*/
			this->resizeBuffer(other.getShape(), other.memoryBuffer.element_size);

			const size_t dataSizeInBytes = other.getDatSize();
			std::memcpy(this->memoryBuffer.buffer.data, other.memoryBuffer.buffer.data, dataSizeInBytes);
			this->typeinfo = other.typeinfo;

			return *this;
		}

		auto &operator=(Tensor &&other) noexcept {
			this->release();

			/*	*/
			this->memoryBuffer.buffer.data = std::exchange(other.memoryBuffer.buffer.data, nullptr);
			this->shape = std::move(other.shape);

			/*	*/
			this->NrElements = other.NrElements;
			this->memoryBuffer.allocationSize = other.memoryBuffer.allocationSize;
			this->memoryBuffer.nrReferences.store(other.memoryBuffer.nrReferences.load());
			this->memoryBuffer.element_size = other.memoryBuffer.element_size;
			this->memoryBuffer.memoryShape = std::move(other.memoryBuffer.memoryShape);

			// TODO: fix own.
			this->memoryBuffer.ownerUid = other.memoryBuffer.ownerUid;
			this->memoryBuffer.uid = other.memoryBuffer.uid;

			this->typeinfo = other.typeinfo;
			return *this;
		}

		Tensor copy() const { return *this; }

		bool operator==(const Tensor &tensor) const noexcept {

			/*	Same address => equal.	*/
			if (static_cast<const void *>(&tensor) == static_cast<const void *>(this)) {
				return true;
			}

			/*	Check internal buffer.	*/
			if (static_cast<const void *>(&tensor.memoryBuffer.buffer.data) ==
				static_cast<const void *>(this->memoryBuffer.buffer.data)) {
				return true;
			}

			/*	Check same shape.	*/
			if (this->getNrElements() != tensor.getNrElements() || this->getDatSize() != tensor.getDatSize()) {
				return false;
			}

			/*	*/
			if (this->getShape() != tensor.getShape()) {
				return false;
			}

			/*	*/
			if (tensor.memoryBuffer.buffer.data == nullptr || this->memoryBuffer.buffer.data == nullptr) {
				return false;
			}

			/*	*/
			if (tensor.typeinfo != this->typeinfo) {
				return false;
			}

			// Last check, see if the content matches.
			if (memcmp(this->memoryBuffer.buffer.data, tensor.memoryBuffer.buffer.data, this->getDatSize()) != 0) {
				return false;
			}

			return true;
		}

		bool operator!=(const Tensor &tensor) const { return !(*this == tensor); }

		// Dtype
		const std::type_info &getDType() const noexcept { return *this->typeinfo; }

		auto &operator-=(const Tensor &tensor) {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) - tensor.getValue<DType>(index);
			}
			return *this;
		}

		auto &operator+=(const Tensor &tensor) {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + tensor.getValue<DType>(index);
			}
			return *this;
		}

		auto &operator*=(const Tensor &tensor) {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) * tensor.getValue<DType>(index);
			}
			return *this;
		}

		/**
		 * @brief Get the Value object
		 */
		template <typename U = DType> inline U getValue(const std::vector<IndexType> &location) const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");
			const size_t index = this->computeShape2Index(location);
			return Tensor::getValue<U>(static_cast<IndexType>(index));
		}

		/**
		 * @brief Get the Value object
		 */
		template <typename U = DType> inline U &getValue(const std::vector<IndexType> &location) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");
			const size_t index = this->computeShape2Index(location);
			return Tensor::getValue<U>(static_cast<IndexType>(index));
		}

		/**
		 * @brief Get the Value object
		 */
		template <typename U = DType> inline U &getValue(const IndexType index) noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");

			const IndexType acuIndex = Shape<IndexType>::getIndexMemoryOffset(this->getShape(), index);

			assert(acuIndex < this->getNrElements());
			U *addr =
				reinterpret_cast<U *>(&this->memoryBuffer.buffer.data[acuIndex * this->memoryBuffer.element_size]);
			return *addr;
		}

		/**
		 * @brief Get the Value object
		 */
		template <typename U = DType> inline U getValue(const IndexType index) const noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");

			const IndexType acuIndex = Shape<IndexType>::getIndexMemoryOffset(this->getShape(), index);

			assert(acuIndex < this->getNrElements());
			const U *addr = reinterpret_cast<const U *>(
				&this->memoryBuffer.buffer.data[acuIndex * this->memoryBuffer.element_size]);
			return *addr;
		}

		/**
		 * @brief Get the Value object
		 */
		template <typename U = DType> inline U *getValuePtr(const IndexType index) const noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");

			const IndexType acuIndex = Shape<IndexType>::getIndexMemoryOffset(this->getShape(), index);

			assert(acuIndex < this->getNrElements());
			U *addr =
				reinterpret_cast<U *>(&this->memoryBuffer.buffer.data[acuIndex * this->memoryBuffer.element_size]);
			return addr;
		}

		auto &operator+(const Tensor &tensor) {

			if (verifyShape(*this, tensor)) {
				/*	*/
			}

			const size_t nrElements = this->getNrElements();
#pragma omp parallel for simd shared(tensor, nrElements)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) + tensor.getValue<DType>(index);
			}
			return *this;
		}

		friend Tensor &operator+(const DType value, Tensor &tensor) noexcept {
			const size_t nrElements = tensor.getNrElements();

#pragma omp simd simdlen(4)
			for (size_t index = 0; index < nrElements; index++) {
				tensor.getValue<DType>(index) = value + tensor.getValue<DType>(index);
			}
			return tensor;
		}

		auto &operator-() noexcept {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for simd simdlen(4)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = -this->getValue<DType>(index);
			}

			return *this;
		}

		auto operator-() const noexcept {
			Tensor output(getShape());
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(output)
			for (size_t index = 0; index < nrElements; index++) {
				output.getValue<DType>(index) = -this->getValue<DType>(index);
			}

			return output;
		}

		auto &operator-(const Tensor &tensor) noexcept {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) - tensor.getValue<DType>(index);
			}

			return *this;
		}

		Tensor operator-(const Tensor &tensor) const noexcept {
			Tensor tmp = Tensor(tensor.getShape());
			size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				tmp.getValue<DType>(index) = this->getValue<DType>(index) - tensor.getValue<DType>(index);
			}

			return tmp;
		}

		friend Tensor &operator-(const DType value, Tensor &tensor) noexcept {
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				tensor.getValue<DType>(index) = value - tensor.getValue<DType>(index);
			}
			return tensor;
		}

		friend Tensor operator-(const DType value, const Tensor &tensor) noexcept {
			const size_t nrElements = tensor.getNrElements();
			Tensor tmp = tensor;
#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				tmp.getValue<DType>(index) = value - tensor.getValue<DType>(index);
			}
			return tmp;
		}

		auto &operator*(const Tensor &tensor) noexcept {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) * tensor.getValue<DType>(index);
			}

			return *this;
		}

		friend auto operator*(const Tensor &tensorA, const Tensor &tensorB) {

			// TODO: verify shape

			Tensor output(tensorA.getShape());
			const size_t nrElements = tensorA.getNrElements();

#pragma omp parallel for shared(tensorA, tensorB, output)
			for (size_t index = 0; index < nrElements; index++) {
				output.getValue<DType>(index) = tensorA.getValue<DType>(index) * tensorB.getValue<DType>(index);
			}

			return output;
		}

		template <typename U> Tensor &operator*(const U vec) noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Type Must Support addition operation.");

			const size_t nrElements = this->getNrElements();

#pragma omp parallel for simd
			for (size_t index = 0; index < nrElements; index++) {
				const DType value = this->getValue<DType>(index);
				const DType result = value * vec;
				this->getValue<DType>(index) = result;
			}

			return *this;
		}

		template <typename U> Tensor operator*(const U vec) const noexcept {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Type Must Support addition operation.");
			Tensor tmp = *this;

			const size_t nrElements = this->getNrElements();

#pragma omp parallel for simd
			for (size_t index = 0; index < nrElements; index++) {
				const DType value = this->getValue<DType>(index);
				const DType result = value * vec;
				tmp.getValue<DType>(index) = result;
			}

			return tmp;
		}

		auto inline &operator%(const Tensor &tensor) noexcept {
			*this = std::move(matrixMultiply(*this, tensor));
			return *this;
		}

		friend inline auto operator%(const Tensor &tensorA, const Tensor &tensorB) {
			return matrixMultiply(tensorA, tensorB);
		}

		template <typename U> auto &operator/(const Tensor &tensor) {
			const size_t nrElements = this->getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t index = 0; index < nrElements; index++) {
				this->getValue<DType>(index) = this->getValue<DType>(index) / tensor.getValue<DType>(index);
			}

			return *this;
		}

		Tensor &assign(const Tensor &other) {
			/*	Transfer data.	*/
			assert(other.getShape().getNrElements() == this->getShape().getNrElements());

			const size_t dataSizeInBytes = other.getDatSize();
			std::memcpy(this->memoryBuffer.buffer.data, other.memoryBuffer.buffer.data, dataSizeInBytes);

			return *this;
		}

		template <typename U = DType> void assignInitValue(const U initValue) noexcept {
			const IndexType nrElements = this->getNrElements();

#pragma omp parallel for simd shared(nrElements, initValue)
			for (size_t i = 0; i < nrElements; i++) {
				this->getValue<DType>(i) = static_cast<DType>(initValue);
			}
		}

		/**
		 * @brief Get the Subset object
		 */
		Tensor getSubset(size_t start, size_t end, const Shape<IndexType> &newShape) const {

			// TODO update shape
			if (newShape.getNrDimensions() == 0) {
				throw InvalidArgumentException("Must Have a valid Shape");
			}
			if ((end - start) != newShape.getNrElements()) {
				throw InvalidArgumentException("Invalid Start/End and Shape");
			}

			//	this->memoryBuffer.nrReferences.fetch_add(1);
			Tensor subset = Tensor(static_cast<uint8_t *>(&this->memoryBuffer.buffer.data[start * DTypeSize]),
								   end - start, newShape, this->memoryBuffer.element_size);

			return subset;
		}

		/**
		 * @brief Get the Subset object
		 */
		Tensor getSubset(const std::initializer_list<ShapePair<IndexType>> subaxisGroup) const {

			const Shape<IndexType> newShape = getShape().getSubShape(subaxisGroup);
			Shape<IndexType> offsetShape = getShape().getSubShape(subaxisGroup);
			offsetShape[0] = (*subaxisGroup.begin()).start0;

			/*	Validate new subset shape size request.	*/
			if (newShape.getNrDimensions() == 0) {
			}

			/*	Create a tensor, parent to this.	*/
			return Tensor(newShape, offsetShape, *this);
		}

		Tensor &flatten() noexcept {
			/*	flatten dim.	*/
			this->shape = {static_cast<IndexType>(this->getNrElements())};
			return *this;
		}

		Tensor &reduce() noexcept {
			/*	flatten dim.	*/
			this->shape.reduce();
			return *this;
		}

		Tensor &reduce(int axis) noexcept {
			/*	flatten dim.	*/
			this->shape.reduce();
			return *this;
		}

		template <typename U> Tensor &insert(const U tensor) { return *this; }

		inline DType dot(const Tensor &tensor) const noexcept { return Tensor::dot(*this, tensor); }

		static inline DType dot(const Tensor &tensorA, const Tensor &tensorB) noexcept {
			// TODO: determine type.
			return Math::dot<DType>(tensorA.getRawData<DType>(), tensorB.getRawData<DType>(), tensorA.getNrElements());
		}

		void dot(const Tensor &tensorB, Tensor &output) const { Tensor::dot(*this, tensorB, output); }

		static Tensor &dot(const Tensor &tensorA, const Tensor &tensorB, Tensor &output) {

			// TODO: determine type.
			const IndexType a0 = tensorB.getShape()[0];
			const IndexType b0 = tensorA.getShape()[0];

			output = std::move(Tensor::zero(Shape<IndexType>({b0, a0}))); // TODO: remove zero when prop impl.

			for (size_t i = 0; i < b0; i++) {
				output.getValue<DType>(i * b0) = Math::dot(tensorB.getValuePtr(a0 * i), tensorA.getValuePtr(0), b0);
			}

			return output;
		}

		Tensor &pow(const DType value) noexcept {
			Math::pow(value, this->getRawData<DType>(), this->getNrElements());
			return *this;
		}

		/**
		 * @brief
		 */
		inline DType mean() const noexcept { return Tensor::mean<DType>(*this); }

		/**
		 * @brief
		 *
		 */
		inline Tensor mean(int axis) noexcept { return Tensor::mean(*this, axis); }

		DType min() const noexcept {
			DType minValue = std::numeric_limits<DType>::max();
			const IndexType elements = this->getNrElements();

#pragma omp parallel for default(shared) reduction(min : minValue)
			for (IndexType i = 0; i < elements; i++) {
				minValue = Math::min<DType>(getValue(i), minValue);
			}
			return minValue;
		}

		/**
		 * @brief
		 *
		 * @return DType
		 */
		DType max() const noexcept {
			DType maxValue = std::numeric_limits<DType>::min();

			const IndexType elements = this->getNrElements();

#pragma omp parallel for default(shared) reduction(max : maxValue)
			for (IndexType i = 0; i < elements; i++) {
				maxValue = Math::max<DType>(this->getValue(i), maxValue);
			}

			return maxValue;
		}

		Tensor &concatenate(const Tensor &tensor, int axis = -1) { // TODO: add axis

			/*	Resize.	*/
			const size_t original_address_offset = this->getDatSize();
			const Shape<IndexType> newShape = tensor.getShape().append(this->getShape(), axis);

			/*	*/
			const size_t element_size = this->getElementSize();
			bool cast = false;
			if (tensor.memoryBuffer.element_size != element_size) {
				// TODO: cast
				cast = true;
			}

			/*	*/
			this->resizeBuffer(newShape, element_size);

			/*	Add additional data.	*/
			if (!cast) {

				// TODO: check memory alignment.
				/*	Copy memory directly.	*/
				const size_t dataSize = tensor.getDatSize();
				uint8_t *dest = this->getRawData<uint8_t>();
				std::memcpy(&dest[original_address_offset], tensor.getRawData<const uint8_t>(), dataSize);
			} else {
				// TODO: impl cast
			}

			return *this;
		}

		template <typename U> Tensor &concatenate(const U value) {
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

		template <typename U> Tensor &append(const std::initializer_list<U> value) {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			/*	Resize.	*/
			this->shape = this->shape + Shape<IndexType>({value.size()});

			/*	Add additional data.	*/
			this->resizeBuffer(this->shape, DTypeSize);

			/*	Copy new Data.	*/ // TODO:impl
			// this->getValue<DType>(this->getNrElements() - 1) = value;

			return *this;
		}

		template <typename U> Tensor<U> &cast() {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");

			/*	*/
			const size_t cast_element_size = sizeof(U);
			if (*this->typeinfo == typeid(U)) { /*	Same type already.	*/
				return reinterpret_cast<Tensor<U> &>(*this);
			}

			/*	Resize.	*/
			if (this->memoryBuffer.element_size != cast_element_size) {

				Tensor<U> tmp = Tensor<U>(this->getShape());

				// #pragma omp parallel for simd
				for (size_t i = 0; i < this->getNrElements(); i++) {
					const DType dvalue = this->getValue<DType>(i);
					tmp.template getValue<U>(i) = static_cast<U>(dvalue);
				}

				Tensor<U> &ref = reinterpret_cast<Tensor<U> &>(*this);
				ref = std::move(tmp);
				return ref;
			}

			/*	*/
			this->memoryBuffer.element_size = cast_element_size;
#pragma omp parallel for simd
			for (size_t i = 0; i < this->getNrElements(); i++) {
				this->getValue<U>(i) = static_cast<U>(this->getValue<DType>(i));
			}

			Tensor<U> &ref = reinterpret_cast<Tensor<U> &>(*this);
			// Convert value.
			return ref;
		}

		template <typename U> Tensor cast() const {
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			const size_t cast_element_size = sizeof(U);

			if (*this->typeinfo == typeid(U)) {
				return reinterpret_cast<Tensor<U> &>(*this);
			}

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

			if (this->memoryBuffer.buffer.data != nullptr && !this->ownAllocation()) {
				/*	*/
				throw RuntimeException("Can not allocate on not owned tensor.");
			}

			if (shape.getNrDimensions() <= 0 || total_nr_elements <= 0) {
				throw RuntimeException("Must be greater than 0");
			}

			/*	Compute size in bytes, aligned.	*/
			const size_t nrByteUnAligned = total_nr_elements * elementSize;
			const size_t nrBytesAllocateAligned = Math::align<size_t>(nrByteUnAligned, alignmentByte);

			// TODO handle if not the same pointer is returned.

			/*	Set ownership if never allocated before.	*/
			if (this->memoryBuffer.buffer.ddata == nullptr) {
				this->memoryBuffer.uid = reinterpret_cast<size_t>(this);
				if (this->memoryBuffer.ownerUid == 0) {
					this->memoryBuffer.ownerUid = this->memoryBuffer.uid;
				}
			}

			/*	*/
			this->memoryBuffer.buffer.data =
				static_cast<uint8_t *>(realloc(this->memoryBuffer.buffer.data, nrBytesAllocateAligned));

			this->memoryBuffer.allocationSize = nrBytesAllocateAligned;

			if (this->memoryBuffer.buffer.data == nullptr) {
				throw RuntimeException("Error");
			}

			this->shape = shape;
			this->NrElements = total_nr_elements;
			this->memoryBuffer.element_size = elementSize;
			this->memoryBuffer.memoryShape = shape;
		}

		friend std::ostream &operator<<(std::ostream &stream, const Tensor &tensor) noexcept {

			const IndexType number_elements = tensor.getNrElements();

			/*	*/
			for (IndexType index = 0; index < number_elements; index++) {

				DType value = tensor.getValue<DType>(index);

				stream << value << ",";
			}

			return stream;
		}

		inline size_t computeShape2Index(const std::vector<IndexType> &dim) const noexcept {
			return Shape<IndexType>::computeIndex(dim, this->shape);
		}

		/**
		 *
		 */
		template <typename U> inline constexpr const U *getRawData() const noexcept {
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");
			return reinterpret_cast<const U *>(this->memoryBuffer.buffer.data);
		}

		/**
		 *
		 */
		template <typename U> inline constexpr U *getRawData() noexcept {
			static_assert(!std::is_pointer<U>::value, "Can not be pointer");
			return reinterpret_cast<U *>(this->memoryBuffer.buffer.data);
		}

		const Shape<IndexType> &getShape() const noexcept { return this->shape; }

		inline IndexType getNrElements() const noexcept { return this->NrElements; }
		inline IndexType getDatSize() const noexcept { return this->getNrElements() * this->memoryBuffer.element_size; }
		inline IndexType getInternalDatSize() const noexcept { return this->memoryBuffer.allocationSize; }
		inline uint32_t getElementSize() const noexcept { return this->memoryBuffer.element_size; }

		static bool verifyShape(const Tensor &tensorA, const Tensor &tensorB) noexcept {
			/*	*/
			return tensorA.getShape() == tensorB.getShape();
		}

		Tensor &reshape(const Shape<IndexType> &newShape) {
			this->shape.reshape(newShape);
			return *this;
		}

	  protected:
		using TensorBuffer = union _buffer_t {
			uint8_t *data = nullptr; /*	*/
			DType *ddata;			 /*	*/
		};

		using InternalBuffer = struct internal_buffer_t {
			TensorBuffer buffer;				 /*	*/
			size_t allocationSize = 0;			 /*	*/
			size_t uid = 0;						 /*	*/
			std::atomic_int32_t nrReferences{1}; /*	*/
			size_t ownerUid = 0;				 /*	*/
			uint32_t element_size = 0;			 /*	*/
			Shape<IndexType> memoryShape;		 /*	*/
		};

		size_t NrElements = 0;			/*	Cache value of shape number of elements.*/
		Shape<IndexType> shape;			/*	Shape of tensor.	*/
		InternalBuffer memoryBuffer;	/*	Internal buffer.	*/
		const std::type_info *typeinfo; /*	*/

		inline bool ownAllocation() const noexcept { return this->memoryBuffer.uid == this->memoryBuffer.ownerUid; }

	  public: /*	Tensor Math Functions.	*/ // TODO: relocate
		static Tensor log10(const Tensor &tensorA) {
			Tensor output(tensorA.getShape());

#pragma omp parallel for shared(tensorA, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<DType>(i) = static_cast<DType>(std::log10(tensorA.getValue<DType>(i)));
			}
			return output;
		}

		static Tensor &abs(Tensor &tensorA) noexcept {

#pragma omp parallel for shared(tensorA)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				tensorA.getValue<Tensor::DType>(i) =
					static_cast<Tensor::DType>(Math::abs<DType>(tensorA.getValue<Tensor::DType>(i)));
			}
			return tensorA;
		}

		static Tensor abs(const Tensor &tensorA) noexcept {
			Tensor output(tensorA.getShape());

#pragma omp parallel for shared(tensorA, output)
			for (size_t i = 0; i < tensorA.getNrElements(); i++) {
				output.getValue<Tensor::DType>(i) =
					static_cast<Tensor::DType>(Math::abs<DType>(tensorA.getValue<Tensor::DType>(i)));
			}
			return output;
		}

		template <typename U> static U mean(const Tensor &tensorA) noexcept {

			/*	*/
			if (tensorA.getNrElements() == 0) {
				return 0;
			}

			return static_cast<U>(Math::mean<DType>(tensorA.getRawData<DType>(), tensorA.getNrElements()));
		}

		static Tensor mean(const Tensor &tensorA, int axis) noexcept {

			/*	*/
			if (tensorA.getNrElements() == 0) {
				return {};
			}

			IndexType dim;
			size_t dim_size;
			if (tensorA.getShape().getNrDimensions() == 1) {
				dim = 1;
				dim_size = 1;
			} else {
				dim = tensorA.getShape().getAxisDimensions(axis);
				dim_size = tensorA.getShape().getAxisDimensions(axis);
			}

			Tensor result({dim});

			/*	*/
			for (size_t i = 0; i < dim_size; i++) {
				const Tensor subset = tensorA.getSubset(
					{{static_cast<IndexType>(i)},
					 {0, tensorA.getShape()[0] - 1} /*TODO:remove*/}); // TODO:fix a unit test to make it work.

				const DType *data = subset.getRawData<DType>();

				const size_t elements = subset.getNrElements();
				const DType meanResult = Math::mean<DType>(data, elements);
				result.getValue(i) = meanResult; // TODO:
			}

			return result;
		}

		template <typename U> static U variance(const Tensor &tensorA, const U mean) noexcept {

			return static_cast<U>(Math::variance<DType>(tensorA.getRawData<DType>(), tensorA.getNrElements(), mean));
		}

		static Tensor variance(const Tensor &tensorA, const DType mean,
							   int axis) noexcept { // TODO: impl and override the none-axis
			// return static_cast<DType>(Math::variance<DType>(tensorA.getRawData<DType>(), tensorA.getNrElements(),
			// mean));
		}

		static Tensor zero(const Shape<IndexType> &shape) {
			Tensor zeroTensor(shape);

			/*	Zero out memory.	*/
			std::memset(zeroTensor.getRawData<void>(), 0, zeroTensor.getInternalDatSize());

			return zeroTensor;
		}

		/**
		 * @brief
		 */
		static Tensor oneShot(const Shape<IndexType> &shape, size_t value) {

			Tensor tensor = std::move(Tensor::zero(shape)); // TODO: improved performance.
			tensor.getValue(value) = static_cast<DType>(1);

			return tensor;
		}

		static Tensor oneShot(const Tensor &tensor, int axis = -1) {

			Shape<IndexType> newShape = tensor.getShape();
			/*	Max values.	*/
			const IndexType max = tensor.max();

			const IndexType maxInclusive = max + 1;
			if (newShape.getNrDimensions() > 1) {
				newShape[-1] = maxInclusive;
			} else {
				newShape.insert(1, Shape<IndexType>({maxInclusive}));
			}

			Tensor<DType> oneshot = std::move(Tensor::zero(newShape));

			/*	*/
			for (IndexType i = 0; i < tensor.getShape()[0]; i++) {
				const IndexType value = tensor.getValue(i);
				oneshot.getSubset({{static_cast<IndexType>(i)}}).getValue(static_cast<IndexType>(value)) = 1;
			}

			/*	Construct.	*/
			return oneshot;
		}

		static Tensor identityMatrix(const Shape<IndexType> &shape) {
			// TODO:verify shape.
			if (shape.getNrDimensions() < 2) {
				throw RuntimeException("Invalid Shape");
			}
			if (shape[0] != shape[1]) {
				throw RuntimeException("Invalid Shape");
			}

			Tensor tensor = std::move(Tensor::zero(shape)); // TODO: improved performance.

			for (IndexType i = 0; i < shape.getAxisDimensions(0); i++) {
				tensor.getValue({i, i}) = 1;
			}

			return tensor;
		}

		static Tensor matrixMultiply(const Tensor &tensorALeft, const Tensor &tensorBRight) {
			/*	TODO: add multiple dims- layers.	*/

			const IndexType B_col = tensorBRight.getShape().getNrDimensions() > 1 ? tensorBRight.getShape()[1] : 1;
			Tensor output(Shape<IndexType>({tensorALeft.getShape()[0], B_col}));

			Tensor::matrixMultiply(tensorALeft, tensorBRight, output);

			return output;
		}

		static constexpr bool isMatrixOperationSupported(const Shape<IndexType> &shapeA,
														 const Shape<IndexType> &shapeB) noexcept {
			const IndexType A_col = shapeA.getNrDimensions() > 1 ? shapeA[1] : 1;
			return A_col == shapeB[0];
		}

		static Tensor matrixMultiply(const Tensor &tensorALeft, const Tensor &tensorBRight, Tensor &output) {

			if (!isMatrixOperationSupported(tensorALeft.getShape(), tensorBRight.getShape())) {
				throw RuntimeException("Invalid Matrix Shape for Multiplication");
			}

			// TODO verify shape.
			if (tensorALeft.getShape().getNrDimensions() > 2) {
				throw NotImplementedException("Not supported");
			}

			const size_t A_row = tensorALeft.getShape()[0];
			const size_t A_col = tensorALeft.getShape().getNrDimensions() > 1 ? tensorALeft.getShape()[1] : 1;

			const size_t B_col = tensorBRight.getShape().getNrDimensions() > 1 ? tensorBRight.getShape()[1] : 1;
			const size_t B_row = tensorBRight.getShape()[0];

			const size_t output_row = output.getShape()[0];
			const size_t output_col = output.getShape().getNrDimensions() > 1 ? output.getShape()[1] : 1;

#pragma omp parallel for collapse(2) shared(tensorALeft, tensorBRight, output)
			for (size_t a_row = 0; a_row < A_row; a_row++) {

				for (size_t b_col = 0; b_col < B_col; b_col++) {

					const size_t indexOutput = a_row * output_col + b_col;

					DType sum = 0;
#pragma omp simd reduction(+ : sum) simdlen(4)
					for (size_t i = 0; i < B_row; i++) {

						const size_t indexA = a_row * A_row + i;
						const size_t indexB = i * B_col + b_col;

						assert(indexA < tensorALeft.getNrElements());
						assert(indexB < tensorBRight.getNrElements());

						sum += tensorALeft.getValue<DType>(indexA) * tensorBRight.getValue<DType>(indexB);
					}

					assert(indexOutput < output.getNrElements());
					output.getValue<DType>(indexOutput) = sum;
				}
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
			static_assert(std::is_floating_point<U>::value || std::is_integral<U>::value,
						  "Must be a decimal type(float/double/half) or integer.");
			Tensor<DType> tensor({static_cast<IndexType>(list.size())});

			IndexType index = 0;
			// #pragma omp simd simdlen(4)
			for (typename std::initializer_list<U>::const_iterator i = list.begin(); i != list.end(); i++) {
				const U value = static_cast<U>(*i);

				tensor.template getValue<DType>(index++) = static_cast<DType>(value);
			}

			return tensor;
		}

		template <typename U> static Tensor split(Tensor &list) { return Tensor({1}, sizeof(U)); }
	};
} // namespace Ritsu