# MiniMax H3 本地 Diffusers 推理

默认使用两张 A100，并从以下本地目录读取 Diffusers 格式权重：

```text
/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3
```

推理不需要 vLLM、不启动 HTTP 服务，也不会访问 Hugging Face Hub。

## 环境要求

当前 Python 环境必须包含支持 MiniMax-H3 的 Modular Diffusers。先检查：

```bash
python -c "from diffusers import ModularPipeline; from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks; print('MiniMax-H3 Diffusers OK')"
```

如果导入失败，说明当前 Diffusers 太旧。请在独立环境安装包含 MiniMax-H3 的新版 Diffusers。为了保护已有 PyTorch，安装本地 Diffusers 源码时使用：

```bash
python -m pip install --no-deps -e /path/to/diffusers
```

不要在已有 VideoX-Fun/Wan 环境里安装 vLLM，也不要让 pip 自动替换 PyTorch。

权重目录至少应包含：

```text
MiniMax-H3/
├── model_index.json
├── transformer/
├── text_encoder/
├── tokenizer/
├── processor/
├── vae/
├── audio_vae/
├── scheduler/
└── audio_scheduler/
```

这里使用的是根目录下的 Diffusers 格式权重，不是 `FL2VA/` 中供 SGLang/vLLM 使用的原始格式。

## 交互式推理

```bash
git pull origin main
CUDA_VISIBLE_DEVICES=2,3 bash run_h3_local.sh
```

模型加载完成后会显示：

```text
Prompt>
```

粘贴一行描述并回车。结果保存到 `outputs/`。可以继续输入下一条；空行或 Ctrl-D 退出。

等价的直接命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 python infer_h3_diffusers.py
```

## 单次推理

```bash
CUDA_VISIBLE_DEVICES=2,3 python infer_h3_diffusers.py \
  --prompt "A cinematic aerial shot of a lighthouse during a storm, synchronized thunder and ocean ambience." \
  --output outputs/h3_test.mp4 \
  --width 1024 --height 576 \
  --duration 5 --steps 50 --seed 42
```

也支持文本文件：

```bash
CUDA_VISIBLE_DEVICES=2,3 python infer_h3_diffusers.py \
  --prompt-file prompt.txt \
  --output outputs/h3_test.mp4
```

## 双卡分配

脚本采用官方 Diffusers 的双卡拆分：

- `cuda:1`：Qwen3-VL 文本编码器；
- `cuda:0`：H3 Transformer、视频 VAE、音频 VAE；
- 两侧均使用 BF16；
- `ComponentsManager` 负责必要时的 CPU offload。

两张 A100 80GB 是推荐配置。两张 A100 40GB 无法直接容纳两个约 62GB 的 BF16 主组件，需要 INT8 或流式 block offload。脚本默认会提前停止，而不是加载到一半 OOM。若只是想尝试 CPU offload，可显式传入：

```bash
CUDA_VISIBLE_DEVICES=2,3 bash run_h3_local.sh --allow-smaller-gpus
```

## 参数覆盖

```bash
CUDA_VISIBLE_DEVICES=2,3 \
WIDTH=1344 HEIGHT=768 DURATION=8 STEPS=50 SEED=42 \
bash run_h3_local.sh
```

H3 固定输出 24 FPS。帧数会自动向上对齐到视频 VAE 支持的 `17*n+5`。

## 常见错误

- `No module named diffusers.modular_pipelines.minimax_h3`：Diffusers 版本太旧。
- `incomplete Diffusers checkpoint`：本地权重缺少根目录 Diffusers 组件。
- 显存小于 70 GiB：很可能是 A100 40GB，需要另做 INT8 路径。
- 找不到 ffmpeg/PyAV：安装视频编码依赖，但不要改动 PyTorch。
