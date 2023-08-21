#include "Ritsu.h"
#include <cmath>
#include <cstdint>

namespace Ritsu {

	template class Layer<std::float_t>;
	template class Layer<std::int32_t>;
	template class Layer<std::int8_t>;

	template class Optimizer<std::float_t>;
	template class Optimizer<double>;

	template class Adam<std::float_t>;
	template class Adam<double>;

	template class SGD<std::float_t>;
	template class SGD<double>;

} // namespace Ritsu