# sudonim
Local AI microservice launcher and docker-compose generator for deploying containerized genAI models and tools.  It provides additional automation for [`jetson-containers`](https://github.com/dusty-nv/jetson-containers) and [`jetson-ai-lab`](https://jetson-ai-lab.com) to manage model downloads, quantization, and serving of OpenAI-compatible endpoints with optimized inference. 

#### Inference APIs

- [x] [MLC](https://github.com/mlc-ai/mlc-llm)
- [x] [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ ] [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

> See https://jetson-ai-lab.com/models.html to use this tool in practice.

## Install

Manual installation isn't typically necessary at this point in time, as normally [`sudonim`](/sudonim) is invoked from docker and already setup inside relevant container images. In jetson-containers it gets installed like this:

```bash
pip install git+https://github.com/dusty-nv/sudonim
```
