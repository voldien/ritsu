
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Valdemar Lindberg
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
#include <cstddef>
#include <stdexcept>
#include <system_error>

namespace Ritsu {
	/*	*/
	using InvalidIndexException = std::runtime_error;
	using RuntimeException = std::runtime_error;
	using DivideByZeroException = std::runtime_error;
	using PermissionDeniedException = std::runtime_error;
	using IOException = std::runtime_error;
	using NotImplementedException = std::runtime_error;
	using InvalidArgumentException = std::runtime_error;
	using NotSupportedException = std::runtime_error;
	using InvalidPointerException = InvalidIndexException;
	using SystemException = std::system_error;

} // namespace Ritsu