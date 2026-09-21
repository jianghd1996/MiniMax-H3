# MiniMax-H3 本地运行

这套脚本使用本地 SGLang 服务运行 H3-Base（768p），默认在下面的目录中自动寻找权重：

```text
/mnt/DataPart/jianghongda/checkpoint
```

> H3-Context-IR 和 H3-Regenerate-2K 没有开源。本地能直接运行的是 H3-Base FL2VA/Ref2VA，默认短边 768。

## 1. 权重目录

脚本会依次检查：

```text
/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3
/mnt/DataPart/jianghongda/checkpoint/minimax-h3
/mnt/DataPart/jianghongda/checkpoint
```

目录至少应为：

```text
MiniMax-H3/
├── model_index.json
├── FL2VA/
│   ├── model_index.json
│   ├── transformer/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── processor/
│   ├── video_vae/
│   └── audio_vae/
└── Ref2VA/                  # 只跑 FL2VA 时可不下载
```

如果实际目录名不同，启动时显式传入：

```bash
MODEL_PATH=/mnt/DataPart/jianghongda/checkpoint/你的目录 \
bash scripts/run_local_server.sh
```

## 2. 环境

建议新建独立环境，不要复用 VideoX-Fun 的环境：

```bash
conda create -n minimax-h3 python=3.11 -y
conda activate minimax-h3

# 按服务器 CUDA 版本安装匹配的 PyTorch，然后安装带 diffusion 支持的 SGLang。
pip install --upgrade pip
pip install "sglang[diffusion]"
```

如果 PyPI 版本还没有 MiniMax-H3 支持，请按仓库 README 指向的
[SGLang MiniMax-H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)
安装对应开发版本。

## 3. 启动 FL2VA 服务

H3 是 33B BF16 模型，默认配置使用 4 张卡：

```bash
cd /path/to/MiniMax-H3

GPU_IDS=0,1,2,3 \
NUM_GPUS=4 \
MODEL_VARIANT=fl2va \
PORT=30010 \
bash scripts/run_local_server.sh
```

脚本默认启用离线加载，避免已经下载好权重后仍访问 Hugging Face。若组件不完整、确实需要联网补文件：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
GPU_IDS=0,1,2,3 NUM_GPUS=4 \
bash scripts/run_local_server.sh
```

显存不足时不要简单降成单卡；应增加 GPU 数量，或依据 SGLang cookbook 使用合适的并行/低显存配置。

## 4. 文生音视频

另开一个终端：

```bash
conda activate minimax-h3
cd /path/to/MiniMax-H3

python scripts/local_generate.py \
  --task t2va \
  --prompt "A cinematic tracking shot of a sailboat crossing a calm sea at sunrise. Gentle waves and natural ocean ambience." \
  --duration 5 \
  --aspect-ratio 16:9 \
  --seed 42 \
  --output outputs/t2va.mp4
```

也可以把长提示词放进文件：

```bash
python scripts/local_generate.py \
  --task t2va \
  --prompt-file prompt.txt \
  --duration 5 \
  --output outputs/t2va.mp4
```

## 5. 首帧生音视频

图像路径由 SGLang 服务进程读取，因此图片必须位于服务器本地且对该进程可见：

```bash
python scripts/local_generate.py \
  --task fl2va \
  --image /absolute/path/to/image.jpg \
  --prompt-file prompt.txt \
  --duration 5 \
  --aspect-ratio auto \
  --seed 42 \
  --output outputs/fl2va.mp4
```

客户端会持续轮询任务状态，成功后才下载视频；失败、超时或服务返回错误时会打印具体原因。

## 6. Ref2VA 服务

参考图像、视频、音频任务需要单独启动 Ref2VA checkpoint，并使用另一个端口：

```bash
GPU_IDS=0,1,2,3 \
NUM_GPUS=4 \
MODEL_VARIANT=ref2va \
PORT=30011 \
bash scripts/run_local_server.sh
```

仓库已有完整 Ref2VA 请求格式示例：
`scripts/readme/reproducible-768p-ref2va-request.sh`。
