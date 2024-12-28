#include "Model.h"
#include "Random.h"
#include "core/Initializers.h"
#include "layers/Layer.h"
#include <cmath>
#include <cstdint>
#include <optimizer/Adam.h>
#include <optimizer/SGD.h>

namespace Ritsu {

	template class RandomNormalInitializer<std::float_t>;

	template class RandomNormal<std::float_t>;

	/*	*/
	template class Shape<std::uint32_t>;
	template class Shape<std::int32_t>;
	template class Shape<std::int16_t>;
	template class Shape<std::uint16_t>;

	/*	*/
	template class Model<std::float_t>;
	// template class Model<std::double_t>;

	/*	*/
	template class Layer<std::float_t>;
	template class Layer<std::int32_t>;
	template class Layer<std::int8_t>;

	// template class Dense<std::float_t>;
	// template class Dense<std::int32_t>;
	// template class Dense<std::int8_t>;

	/*	*/
	template class Tensor<std::double_t>;
	template class Tensor<std::float_t>;
	template class Tensor<std::int32_t>;
	template class Tensor<std::int16_t>;
	template class Tensor<std::int8_t>;
	template class Tensor<bool>;

	/*	*/
	template class Optimizer<std::float_t>;
	template class Optimizer<std::double_t>;

	/*	*/
	template class Adam<std::float_t>;
	template class Adam<std::double_t>;

	/*	*/
	template class SGD<std::float_t>;
	template class SGD<std::double_t>;

} // namespace Ritsu