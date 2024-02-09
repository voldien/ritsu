#include <Ritsu.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>

namespace RitsuDataSet {

	extern void loadMNIST(const std::string &imagePath, const std::string &labelPath, const std::string &imageTestPath,
						  const std::string &labelTestPath, Ritsu::Tensor<float> &dataX, Ritsu::Tensor<uint8_t> &dataY,
						  Ritsu::Tensor<float> &testX, Ritsu::Tensor<uint8_t> &testY);

} // namespace RitsuDataSet