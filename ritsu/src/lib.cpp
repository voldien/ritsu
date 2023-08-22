#include "Ritsu.h"
#include <cmath>
#include <cstdint>

namespace Ritsu {

	template class Model<std::float_t>;
	template class Model<std::int32_t>;
	template class Model<std::int8_t>;

	template class Layer<std::float_t>;
	template class Layer<std::int32_t>;
	template class Layer<std::int8_t>;

	//template class Tensor<std::float_t>;
	//template class Tensor<std::int32_t>;
	//template class Tensor<std::int8_t>;

	template class Optimizer<std::float_t>;
	template class Optimizer<double>;

	template class Adam<std::float_t>;
	template class Adam<double>;

	template class SGD<std::float_t>;
	template class SGD<double>;

} // namespace Ritsu