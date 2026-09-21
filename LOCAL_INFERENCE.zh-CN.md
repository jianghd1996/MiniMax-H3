# MiniMax H3 本地推理

这套脚本从本地权重目录加载 H3，不会在启动时从 Hugging Face 下载模型。默认路径已经设为：

```text
/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3
```

当前先提供最容易验证的 FL2VA 服务（文生视频与首/尾帧模型分区）和 T2VA 客户端。输出是带 32 kHz 立体声音频的 MP4。

## 1. 检查权重结构

至少应存在：

```text
MiniMax-H3/
├── model_index.json
├── FL2VA/
├── text_encoder/
├── tokenizer/
├── processor/
├── vae/
└── audio_vae/
```

若目录里只有若干 safetensors、缺少配置文件或 `FL2VA/`，启动脚本会提前报错。

## 2. 安装环境

需要 Linux、CUDA、Git、ffmpeg，以及足够大的系统内存。H3 单个任务分区的 BF16 权重约 135 GiB；单卡方式依赖 CPU offload，并不适合普通 24 GB 显卡配合小内存主机。

```bash
cd MiniMax-H3
bash scripts/local/setup_vllm_omni.sh
source .venv-h3/bin/activate
```

脚本按官方 vLLM-Omni 配方安装 `vllm==0.26.0` 和当前 vLLM-Omni 源码。

## 3. 启动服务

按硬件选择一个配置。

单卡（CPU offload，需要大量内存）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/local/serve_h3.sh single
```

双 RTX 4090（建议先从 1024×576、5 秒开始）：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/local/serve_h3.sh 2x4090
```

双 RTX 5090：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/local/serve_h3.sh 2x5090
```

四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/local/serve_h3.sh 4gpu
```

自定义权重或端口：

```bash
MODEL_PATH=/path/to/MiniMax-H3 PORT=30010 \
CUDA_VISIBLE_DEVICES=0,1 bash scripts/local/serve_h3.sh 2x4090
```

看到 `Application startup complete` 后再运行客户端。首次初始化和首次生成都可能很慢。

## 4. 生成测试视频

```bash
python scripts/local/generate_t2va.py \
  --prompt "A cinematic aerial shot of a lighthouse during a storm, synchronized thunder and ocean ambience." \
  --output outputs/h3_test.mp4 \
  --width 1024 --height 576 \
  --duration 5 --steps 50 --seed 42
```

也可把长提示词放进 UTF-8 文本文件：

```bash
python scripts/local/generate_t2va.py \
  --prompt-file prompt.txt \
  --output outputs/h3_test.mp4
```

验证输出：

```bash
ffprobe -v error -show_entries \
  stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json outputs/h3_test.mp4
```

## 常见问题

- `missing FL2VA`：当前目录不是完整的 H3 根 checkpoint。
- 进程被系统直接 kill：通常是系统内存或 pinned memory 不足。
- CUDA OOM：先使用匹配的低显存 profile，并保持 1024×576、5 秒。
- `vllm: command not found`：先执行安装脚本并激活 `.venv-h3`。
- SGLang 的 `--model-variant` 报未知参数：本地脚本不走这条兼容性不稳定的路径。
- 本地开源版本只覆盖 H3-Base；官方 Context-IR 和 Regenerate-2K 不在开源权重内。
