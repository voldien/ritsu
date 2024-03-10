
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023 Valdemar Lindberg
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */
#pragma once
#include "Ritsu.h"
#include "RitsuDef.h"

namespace Ritsu {

	template <typename U> std::ostream &debug_layer(std::ostream &stream, const Layer<U> &layer) noexcept {

		stream << layer.getName() << std::endl << std::endl;

		if (layer.getTrainableWeights()) {
			std::cerr << "trainable: " << *layer.getTrainableWeights() << std::endl << std::endl;
		}

		if (layer.getVariables()) {
			std::cerr << "non-trainable: " << *layer.getVariables() << std::endl << std::endl;
		}

		return stream;
	}
} // namespace Ritsu