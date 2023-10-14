#include <Ritsu.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <istream>
#include <ostream>

namespace RitsuDataSet {

	extern void loadMNIST(const std::string &imagePath, const std::string &labelPath, const std::string &imageTestPath,
						  const std::string &labelTestPath, Ritsu::Tensor &dataX, Ritsu::Tensor &dataY,
						  Ritsu::Tensor &testX, Ritsu::Tensor &testY);

} // namespace RitsuDataSet