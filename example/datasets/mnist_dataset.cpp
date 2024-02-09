#include "mnist_dataset.h"
#include "RitsuDef.h"
#include "core/Shape.h"
#include <cassert>
#include <cstdint>
#include <exception>
#include <fstream>
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

static void load_image_dataset(std::ifstream &stream, Ritsu::Tensor<float> &dataX) {
	/*	*/
	int32_t width, height, nr_images, image_magic;

	/*	*/
	stream.seekg(0, std::ios::beg);

	/*	*/
	stream.read((char *)&image_magic, sizeof(image_magic));
	stream.read((char *)&nr_images, sizeof(nr_images));
	stream.read((char *)&width, sizeof(width));
	stream.read((char *)&height, sizeof(height));

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

	dataX = Tensor<float>(
		{static_cast<unsigned int>(nr_images), static_cast<unsigned int>(width), static_cast<unsigned int>(height), 1},
		sizeof(uint8_t));

	Shape<unsigned int> expected;
	expected = {static_cast<unsigned int>(nr_images), static_cast<unsigned int>(width),
				static_cast<unsigned int>(height), 1};
	assert(dataX.getShape() == expected);

	uint8_t *raw = dataX.getRawData<uint8_t>();

	uint8_t *imageData = (uint8_t *)malloc(ImageSize);

	for (size_t i = 0; i < nr_images; i++) {
		stream.read((char *)&imageData[0], ImageSize);
		// swap value...

		memcpy(&raw[i * ImageSize], imageData, ImageSize);
	}

	free(imageData);
}

static void load_label_dataset(std::ifstream &stream, Ritsu::Tensor<uint8_t> &dataY) {

	uint32_t label_magic, nr_label;

	stream.seekg(0, std::ios::beg);

	/*	*/
	stream.read((char *)&label_magic, sizeof(label_magic));
	stream.read((char *)&nr_label, sizeof(nr_label));

	/*	*/
	label_magic = swap_endian(label_magic);
	nr_label = swap_endian(nr_label);

	const uint32_t magic_number = 0x00000801;

	// Verify
	if (label_magic != magic_number) {
		throw std::runtime_error("Invalid magic number for label training data.");
	}

	dataY = Tensor<uint8_t>({nr_label, 1}, sizeof(uint32_t));

	Shape<unsigned int> expected;
	expected = {static_cast<unsigned int>(nr_label), 1};
	assert(dataY.getShape() == expected);

	uint8_t label;
	for (size_t i = 0; i < nr_label; i++) {

		stream.read((char *)&label, sizeof(label));
		dataY.getValue<uint8_t>(i) = label;
	}
}

void RitsuDataSet::loadMNIST(const std::string &imagePath, const std::string &labelPath,
							 const std::string &imageTestPath, const std::string &labelTestPath,
							 Ritsu::Tensor<float> &dataX, Ritsu::Tensor<uint8_t> &dataY, Ritsu::Tensor<float> &testX,
							 Ritsu::Tensor<uint8_t> &testY) {

	/*	*/
	std::ifstream imageTrainStream(imagePath, std::ios::in | std::ios::binary);
	std::ifstream labelTrainStream(labelPath, std::ios::in | std::ios::binary);

	/*	*/
	std::ifstream imageTestStream(imageTestPath, std::ios::in | std::ios::binary);
	std::ifstream labelTestStream(labelTestPath, std::ios::in | std::ios::binary);

	{
		if (!imageTrainStream.is_open()) {
			throw RuntimeException("Failed to open file image path");
		}

		load_image_dataset(imageTrainStream, dataX);

		if (!imageTestStream.is_open()) {
			throw RuntimeException("Failed to open file image path");
		}
		load_image_dataset(imageTestStream, testX);
	}

	{
		if (!labelTrainStream.is_open()) {
			throw RuntimeException("Failed to open file label path");
		}

		load_label_dataset(labelTrainStream, dataY);

		if (!labelTestStream.is_open()) {
			throw RuntimeException("Failed to open file label path");
		}
		load_label_dataset(labelTestStream, testY);
	}
}