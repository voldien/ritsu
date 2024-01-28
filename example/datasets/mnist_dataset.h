#include <Ritsu.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>

namespace RitsuDataSet {

	extern void loadMNIST(const std::string &imagePath, const std::string &labelPath, const std::string &imageTestPath,
						  const std::string &labelTestPath, Ritsu::Tensor<float> &dataX, Ritsu::Tensor<float> &dataY,
						  Ritsu::Tensor<float> &testX, Ritsu::Tensor<float> &testY);

} // namespace RitsuDataSet