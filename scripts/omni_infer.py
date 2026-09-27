#!/usr/bin/env python3
"""Qwen2.5-Omni 推理脚本

支持文本输入 → 文本+语音输出。
在服务器上运行：
    /root/miniconda3/bin/python scripts/omni_infer.py
    /root/miniconda3/bin/python scripts/omni_infer.py --text "介绍一下自己"
"""
import argparse
import time

MODEL_PATH = "/gemini/pretrain"
DEFAULT_PROMPT = "你好，请介绍一下你自己。"
SAMPLE_RATE = 24000


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni 推理")
    parser.add_argument("--text", type=str, default=DEFAULT_PROMPT, help="用户输入文本")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="系统提示词")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--output-wav", type=str, default="/tmp/reply.wav", help="语音输出路径")
    args = parser.parse_args()

    import torch
    import soundfile as sf
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    # 加载模型
    print("加载模型中（首次约 5-6 分钟）...")
    t0 = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    t1 = time.time()
    used = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"模型加载完成: {t1 - t0:.0f}s | GPU: {used:.1f}GB / {total:.1f}GB")

    # 构建对话
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": args.system}]},
        {"role": "user", "content": [{"type": "text", "text": args.text}]},
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)

    # 推理
    print(f"生成中... (max_new_tokens={args.max_tokens})")
    t2 = time.time()
    gen_output = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        return_audio=True,
    )
    t3 = time.time()

    # 解包 generate 返回值
    if isinstance(gen_output, tuple):
        sequences = gen_output[0]
        audio_wav = gen_output[1] if len(gen_output) > 1 else None
    else:
        sequences = getattr(gen_output, "sequences", gen_output)
        audio_wav = getattr(gen_output, "audios", None)

    # 解码文本
    new_tokens = sequences[:, inputs["input_ids"].shape[1]:]
    text_output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print(f"\n推理耗时: {t3 - t2:.1f}s")
    print("=== 文本回复 ===")
    print(text_output)

    # 保存语音
    if audio_wav is not None:
        if isinstance(audio_wav, (list, tuple)):
            audio_wav = audio_wav[0]
        audio_np = audio_wav.detach().cpu().numpy().squeeze()
        sf.write(args.output_wav, audio_np, samplerate=SAMPLE_RATE)
        duration = audio_np.shape[-1] / SAMPLE_RATE
        print(f"=== 语音已保存: {args.output_wav} ({duration:.1f}s) ===")
    else:
        print("=== 无语音输出 ===")

    print("=== DONE ===")


if __name__ == "__main__":
    main()
