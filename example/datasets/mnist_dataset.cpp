#include "mnist_dataset.h"
#include <cstdint>
#include <exception>
#include <stdexcept>

using namespace Ritsu;

template <typename T> static inline constexpr T swap_endian(const T value) {
	// Swap endian (big to little) or (little to big)

	/*	*/
	const uint32_t bi0 = (value & 0x000000ff) << 24u;
	const uint32_t bi1 = (value & 0x0000ff00) << 8u;
	const uint32_t bi2 = (value & 0x00ff0000) >> 8u;
	const uint32_t bi3 = (value & 0xff000000) >> 24u;

	return bi0 | bi1 | bi2 | bi3;
}

void RitsuDataSet::loadMNIST(const std::string &imagePath, const std::string &labelPath,
							 const std::string &imageTestPath, const std::string &labelTestPath, Ritsu::Tensor<float> &dataX,
							 Ritsu::Tensor<float> &dataY, Ritsu::Tensor<float> &testX, Ritsu::Tensor<float> &testY) {

	/*	*/
	std::ifstream imageTrainStream(imagePath, std::ios::in | std::ios::binary);
	std::ifstream labelTrainStream(labelPath, std::ios::in | std::ios::binary);

	/*	*/
	std::ifstream imageTestStream(imageTestPath, std::ios::in | std::ios::binary);
	std::ifstream labelTestStream(labelTestPath, std::ios::in | std::ios::binary);

	if (!imageTrainStream.is_open()) {
		throw std::runtime_error("Failed to open file image path");
	}

	{
		/*	*/
		int32_t width, height, nr_images, image_magic;

		/*	*/
		imageTrainStream.seekg(0, std::ios::beg);

		/*	*/
		imageTrainStream.read((char *)&image_magic, sizeof(image_magic));
		imageTrainStream.read((char *)&nr_images, sizeof(nr_images));
		imageTrainStream.read((char *)&width, sizeof(width));
		imageTrainStream.read((char *)&height, sizeof(height));

		/*	*/
		image_magic = swap_endian(image_magic);
		nr_images = swap_endian(nr_images);
		width = swap_endian(width);
		height = swap_endian(height);

		const uint32_t magic_number = 0x00000803;

		// Verify
		if (image_magic != magic_number) {
			throw std::runtime_error("Invalid magic number for image training data.");
		}

		const size_t ImageSize = static_cast<size_t>(width) * static_cast<size_t>(height);

		dataX = Tensor<float>({static_cast<unsigned int>(nr_images), static_cast<unsigned int>(width),
						static_cast<unsigned int>(height), 1},
					   sizeof(uint8_t));
		uint8_t *raw = dataX.getRawData<uint8_t>();

		uint8_t *imageData = (uint8_t *)malloc(ImageSize);

		for (size_t i = 0; i < nr_images; i++) {
			imageTrainStream.read((char *)&imageData[0], ImageSize);
			// swap value...

			memcpy(&raw[i * ImageSize], imageData, ImageSize);
		}

		free(imageData);
	}

	if (!labelTrainStream.is_open()) {
		throw std::runtime_error("Failed to open file label path");
	}
	{

		uint32_t label_magic, nr_label;

		labelTrainStream.seekg(0, std::ios::beg);

		/*	*/
		labelTrainStream.read((char *)&label_magic, sizeof(label_magic));
		labelTrainStream.read((char *)&nr_label, sizeof(nr_label));

		/*	*/
		label_magic = swap_endian(label_magic);
		nr_label = swap_endian(nr_label);

		const uint32_t magic_number = 0x00000801;

		// Verify
		if (label_magic != magic_number) {
			throw std::runtime_error("Invalid magic number for label training data.");
		}

		dataY = Tensor<float>({nr_label, 1}, sizeof(uint32_t));
		uint32_t label;

		for (size_t i = 0; i < nr_label; i++) {

			labelTrainStream.read((char *)&label, sizeof(label));
			dataX.getValue<uint32_t>(i) = label;
		}
	}
}