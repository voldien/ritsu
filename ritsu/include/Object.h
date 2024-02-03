#pragma once
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Object {
	  public:
		Object(const std::string &name) noexcept : name(name) {}

		const std::string &getName() const noexcept { return this->name; }
		void setName(const std::string &name) noexcept { this->name = name; }

	  private:
		std::string name;
	};
} // namespace Ritsu