#include "Ritsu.h"
#include <cmath>
#include <cstdint>

namespace Ritsu {

	template class Shape<std::uint32_t>;
	// template class Shape<std::int32_t>;

	template class Model<std::float_t>;
	template class Model<std::int32_t>;
	template class Model<std::int8_t>;

	template class Layer<std::float_t>;
	template class Layer<std::int32_t>;
	template class Layer<std::int8_t>;

	// template class Dense<std::float_t>;
	// template class Dense<std::int32_t>;
	// template class Dense<std::int8_t>;

	template class Tensor<std::double_t>;
	template class Tensor<std::float_t>;
	template class Tensor<std::int32_t>;
	template class Tensor<std::int8_t>;
	template class Tensor<bool>;

	template class Optimizer<std::float_t>;
	template class Optimizer<std::double_t>;

	template class Adam<std::float_t>;
	template class Adam<std::double_t>;

	template class SGD<std::float_t>;
	template class SGD<std::double_t>;

} // namespace Ritsu