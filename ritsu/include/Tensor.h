#pragma once
#include <string>
#include <vector>

namespace Ritsu {

	class Tensor {

	  public:
		Tensor() {}
		Tensor(const std::vector<unsigned int> &dimensions, unsigned int elementSize) {
			resizeBuffer(dimensions, elementSize);
		}

		// operations of data.
		template <typename T> T getValue(const std::vector<unsigned int> &location) const {
			size_t index = 0;

			return static_cast<T>(buffer[index]);
		}

		template <typename T> T &getValue(const std::vector<unsigned int> &location) {
			size_t index = 0;
			T *addr = reinterpret_cast<T*>(&buffer[index]);
			T& ref = (T&)addr;
			return ref;
		}

		//TODO add std::cout istream

		float operator[](const std::vector<unsigned int> &location) const { return getValue<float>(location); }
		float &operator[](const std::vector<unsigned int> &location) { return getValue<float>(location); }

		void resizeBuffer(const std::vector<unsigned int> &dimensions, unsigned int elementSize) {
			long int totalSize = 1;
			for (size_t i = 0; i < dimensions.size(); i++) {
				totalSize *= dimensions[i];
			}
			buffer.resize(totalSize * elementSize);
			this->dimensions = dimensions;
		}

		const std::vector<unsigned int> &getNrDimension() const { return this->dimensions; }

	  private:
		std::vector<unsigned int> dimensions;
		std::vector<uint8_t> buffer;
	};
} // namespace Ritsu