#include "mnist_dataset.h"
#include <exception>
#include <stdexcept>

using namespace Ritsu;

template <typename T> static T swap_endian(T value) {
	// Swap endian (big to little) or (little to big)
	uint32_t b0, b1, b2, b3;
	uint32_t res;

	b0 = (value & 0x000000ff) << 24u;
	b1 = (value & 0x0000ff00) << 8u;
	b2 = (value & 0x00ff0000) >> 8u;
	b3 = (value & 0xff000000) >> 24u;

	return b0 | b1 | b2 | b3;
}

void RitsuDataSet::loadMNIST(const std::string &imagePath, const std::string &labelPath,
							 const std::string &imageTestPath, const std::string &labelTestPath, Ritsu::Tensor &dataX,
							 Ritsu::Tensor &dataY, Ritsu::Tensor &testX, Ritsu::Tensor &testY) {
	/*	*/
	std::ifstream imageStream(imagePath, std::ios::in | std::ios::binary);
	std::ifstream labelStream(labelPath, std::ios::in | std::ios::binary);

	if (!imageStream.is_open()) {
		throw std::runtime_error("Failed to open file image path");
	}

	else {

		/*	*/
		int32_t width, height, nr_images, image_magic;

		imageStream.seekg(0, std::ios::beg);

		imageStream.read((char *)&image_magic, sizeof(image_magic));
		imageStream.read((char *)&nr_images, sizeof(nr_images));
		imageStream.read((char *)&width, sizeof(width));
		imageStream.read((char *)&height, sizeof(height));

		image_magic = swap_endian(image_magic);
		nr_images = swap_endian(nr_images);
		width = swap_endian(width);
		height = swap_endian(height);

		// Verify
		if (image_magic != 0x00000803) {
			throw std::runtime_error("Invalid magic number for image training data.");
		}

		const size_t ImageSize = static_cast<size_t>(width) * static_cast<size_t>(height);

		dataX = Tensor({nr_images, width, height, 1}, sizeof(uint8_t));
		uint8_t *raw = dataX.getRawData<uint8_t>();

		uint8_t *imageData = (uint8_t *)malloc(ImageSize);

		for (size_t i = 0; i < nr_images; i++) {
			imageStream.read((char *)&imageData[0], ImageSize);
			// swap value...

			memcpy(&raw[i * ImageSize], imageData, ImageSize);
		}

		free(imageData);
	}

	if (!labelStream.is_open()) {
		throw std::runtime_error("Failed to open file label path");
	} else {

		uint32_t label_magic, nr_label;

		labelStream.seekg(0, std::ios::beg);
		labelStream.read((char *)&label_magic, sizeof(label_magic));
		labelStream.read((char *)&nr_label, sizeof(nr_label));

		label_magic = swap_endian(label_magic);
		nr_label = swap_endian(nr_label);

		// Verify
		if (label_magic != 0x00000801) {
			throw std::runtime_error("Invalid magic number for label training data.");
		}

		dataY = Tensor({nr_label, 1}, sizeof(uint32_t));
		uint32_t label;
		for (size_t i = 0; i < nr_label; i++) {

			labelStream.read((char *)&label, sizeof(label));
			dataX.getValue<uint32_t>(i) = label;
		}
	}
}