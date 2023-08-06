#pragma once
#include "Layer.h"
#include <ctime>
#include <random>

namespace Ritsu {

	class GuassianNoise : public Layer<float> {

	  public:
		GuassianNoise(float stddev, const std::string &name = "") : Layer(name), stddev(stddev) {
			std::random_device rd;
			this->gen = std::mt19937(rd());
			this->dis = std::uniform_real_distribution<>(1.0, 2.0);
		}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor tmp = tensor;
			this->addNoise(tmp);
			return tmp;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->addNoise(tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
			this->addNoise(tensor);
			return tensor;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		void addNoise(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
#pragma omp simd
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) += dis(gen);
			}
		}

	  private:
		float stddev;
		std::mt19937 gen;
		std::uniform_real_distribution<> dis;
	};
} // namespace Ritsu