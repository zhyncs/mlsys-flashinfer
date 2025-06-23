# mlsys-flashinfer

FlashInfer for MLSys 2025

# Benchmark

Testing online scenarios is primarily latency-sensitive, so ensuring TTFT is crucial. Generally, P99 TTFT should be within 200ms and not exceed 500ms at worst. At this point, the throughput is **goodput**. To meet this requirement, we adjusted the request rate.

## Llama-3.1-8B-Instruct

### ShareGPT Scenario

```bash
# server

## Triton
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache --enable-overlap-schedule --dtype float16 --attention triton

## FlashInfer
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache --enable-overlap-schedule --dtype float16

## TensorRT LLM
## v0.13.0
docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
## fix for Llama 3.1
pip install transformers==4.46.0
python3 convert_checkpoint.py --model_dir /models/Llama-3.1-8B-Instruct --dtype float16 --tp_size 1 --output_dir llama3-8b
trtllm-build --checkpoint_dir=llama3-8b --output_dir=llama3-8b-engine --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --max_batch_size=256 --max_input_len=4096 --max_num_tokens=4096
python3 launch_triton_server.py --world_size=1 --model_repo=/tensorrt-demo/triton_model_repo --http_port 8123

# client

## Triton
python -m sglang.bench_serving --backend sglang --request-rate 46

## FlashInfer
python -m sglang.bench_serving --backend sglang --request-rate 46

## TensorRT LLM
python -m sglang.bench_serving --backend trt --request-rate 46 --model meta-llama/Llama-3.1-8B-Instruct --port 8123
```

### Random Scenario

```bash
# server

## Triton
## not use enable_overlap_schedule
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache --dtype float16 --attention triton

## FlashInfer
## not use enable_overlap_schedule
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache --dtype float16

## TensorRT LLM
## v0.13.0
docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
## fix for Llama 3.1
pip install transformers==4.46.0
python3 convert_checkpoint.py --model_dir /models/Llama-3.1-8B-Instruct --dtype float16 --tp_size 1 --output_dir llama3-8b
trtllm-build --checkpoint_dir=llama3-8b --output_dir=llama3-8b-engine --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --max_batch_size=256 --max_input_len=4096 --max_num_tokens=4096
python3 launch_triton_server.py --world_size=1 --model_repo=/tensorrt-demo/triton_model_repo --http_port 8123

# client

## Triton
python -m sglang.bench_serving --backend sglang --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.25 --request-rate 8 --dataset-name random --num-prompts 256

## FlashInfer
python -m sglang.bench_serving --backend sglang --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.25 --request-rate 8 --dataset-name random --num-prompts 256

## TensorRT LLM
python -m sglang.bench_serving --backend trt --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.25 --request-rate 8 --dataset-name random --num-prompts 256 --model meta-llama/Llama-3.1-8B-Instruct --port 8123
```

## Llama-3.1-70B-Instruct

### ShareGPT Scenario

```bash
# server

## Triton
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --enable-overlap-schedule --dtype float16 --attention triton --tp 4

## FlashInfer
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --enable-overlap-schedule --dtype float16 --tp 4

## TensorRT LLM
## v0.13.0
docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
## fix for Llama 3.1
pip install transformers==4.46.0
python3 convert_checkpoint.py --model_dir /models/Llama-3.1-70B-Instruct --dtype float16 --tp_size 4 --output_dir llama3-70b
trtllm-build --checkpoint_dir=llama3-70b --output_dir=llama3-70b-engine --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --max_batch_size=256 --max_input_len=4096 --max_num_tokens=4096 --workers 4
python3 launch_triton_server.py --world_size=4 --model_repo=/tensorrt-demo/triton_model_repo --http_port 8123

# client

## Triton
python -m sglang.bench_serving --backend sglang --request-rate 16 --num-prompts 512

## FlashInfer
python -m sglang.bench_serving --backend sglang --request-rate 16 --num-prompts 512

## TensorRT LLM
python -m sglang.bench_serving --backend trt --request-rate 16 --model meta-llama/Llama-3.1-70B-Instruct --port 8123 --num-prompts 512
```

### Random Scenario

```bash
# server

## Triton
## not use enable_overlap_schedule
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --dtype float16 --attention triton --tp 4

## FlashInfer
## not use enable_overlap_schedule
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --dtype float16 --tp 4

## TensorRT LLM
## v0.13.0
docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
## fix for Llama 3.1
pip install transformers==4.46.0
python3 convert_checkpoint.py --model_dir /models/Llama-3.1-70B-Instruct --dtype float16 --tp_size 4 --output_dir llama3-70b
trtllm-build --checkpoint_dir=llama3-70b --output_dir=llama3-70b-engine --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --max_batch_size=256 --max_input_len=4096 --max_num_tokens=4096 --workers 4
python3 launch_triton_server.py --world_size=4 --model_repo=/tensorrt-demo/triton_model_repo --http_port 8123

# client

## Triton
python -m sglang.bench_serving --backend sglang --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.25 --request-rate 4 --dataset-name random --num-prompts 256

## FlashInfer
python -m sglang.bench_serving --backend sglang --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.25 --request-rate 4 --dataset-name random --num-prompts 256

## TensorRT LLM
python -m sglang.bench_serving --backend trt --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.25 --request-rate 4 --dataset-name random --num-prompts 256 --model meta-llama/Llama-3.1-70B-Instruct --port 8123
```

# Appendix

- https://github.com/yzh119/flashinfer-dev/tree/hopper

```bash
git clone -b hopper https://github.com/yzh119/flashinfer-dev
cd flashinfer-dev
git submodule update --init --recursive
cd flashinfer-aot
export TORCH_CUDA_ARCH_LIST=9.0+PTX
pip install -e . -v
```

- https://github.com/MasterJH5574/tensorrt-demo

```bash
find triton_model_repo -name "config.pbtxt" -exec sed -i 's/batch_size: 2048/batch_size: 256/g' {} \;
find triton_model_repo -name "config.pbtxt" -exec sed -i 's/preferred_batch_size: \[ 2048 \]/preferred_batch_size: [ 256 ]/g' {} \;
```

- https://github.com/sgl-project/sglang

```bash
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py
# decoding use tensor core
# use ragged forward
# remove q/k/v contiguous for forward_return_lse
```
