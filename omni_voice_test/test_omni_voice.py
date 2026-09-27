import base64
import json
import os
import traceback
import urllib.error
import urllib.request

OUT = r"D:\butheisDelight\omni_voice_test"
MIMO_BASE = "https://api.xiaomimimo.com/v1"
MIMO_KEY = os.environ["MIMO_KEY"]
OMNI_BASE = "http://llm-model-hub-proxy.sit.sf-express.com/v1"
OMNI_KEY = os.environ["OMNI_KEY"]
OMNI_MODEL = "aliyun/qwen3.5-omni-plus"
SPEECH = "你好，我是用小米语音合成发过来的一条语音。请你用音频回复：生成一段简短欢快的中文儿歌或小旋律，时长十秒左右，唱出来或哼出来都可以。"


def post_json(url, headers, payload, timeout=120):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, body
    except Exception as e:
        return -1, str(e).encode("utf-8", errors="replace")


def save_audio_from_message(message, tag):
    """Save audio from various response shapes. Returns path or None."""
    audio = message.get("audio") if isinstance(message, dict) else None
    if not audio:
        # OpenAI SDK style may nest differently
        if isinstance(message, dict):
            for k in ("audio", "output_audio"):
                if message.get(k):
                    audio = message[k]
                    break
    if not audio:
        print(f"[{tag}] no audio object in message keys: {list(message.keys()) if isinstance(message, dict) else type(message)}")
        return None

    data = None
    fmt = "wav"
    if isinstance(audio, dict):
        data = audio.get("data") or audio.get("b64_json") or audio.get("id")
        fmt = audio.get("format") or audio.get("mime_type") or "wav"
    elif isinstance(audio, str):
        data = audio
    if not data:
        print(f"[{tag}] audio object has no data: {audio if not isinstance(audio, dict) else list(audio.keys())}")
        return None
    if isinstance(data, str) and data.startswith("data:"):
        # data:audio/wav;base64,xxxx
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data)
    except Exception as e:
        print(f"[{tag}] b64 decode failed: {e}")
        return None
    ext = "wav"
    if isinstance(fmt, str):
        if "mpeg" in fmt or "mp3" in fmt:
            ext = "mp3"
        elif "flac" in fmt:
            ext = "flac"
        elif "ogg" in fmt:
            ext = "ogg"
        elif "m4a" in fmt or "mp4" in fmt:
            ext = "m4a"
        elif "pcm" in fmt or "raw" in fmt:
            ext = "pcm"
        elif "wav" in fmt:
            ext = "wav"
    path = os.path.join(OUT, f"{tag}.{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    print(f"[{tag}] saved {path} bytes={len(raw)} format={fmt}")
    return path


def summarize_response(status, body, tag):
    print(f"\n===== {tag} status={status} =====")
    try:
        obj = json.loads(body.decode("utf-8"))
    except Exception:
        text = body.decode("utf-8", errors="replace")
        print(f"[{tag}] non-json body (first 800):\n{text[:800]}")
        return None
    print(f"[{tag}] top keys: {list(obj.keys())}")
    # dump compact summary
    if "choices" in obj and obj["choices"]:
        msg = obj["choices"][0].get("message", {})
        print(f"[{tag}] message keys: {list(msg.keys())}")
        content = msg.get("content")
        if content:
            preview = content if isinstance(content, str) else str(content)[:2000]
            print(f"[{tag}] content preview:\n{preview[:2000]}")
        if msg.get("reasoning_content"):
            rc = msg["reasoning_content"]
            print(f"[{tag}] reasoning preview:\n{str(rc)[:800]}")
        print(f"[{tag}] finish_reason: {obj['choices'][0].get('finish_reason')}")
        print(f"[{tag}] usage: {obj.get('usage')}")
        print(f"[{tag}] model: {obj.get('model')}")
        if isinstance(msg.get("audio"), dict) or msg.get("audio"):
            print(f"[{tag}] HAS audio payload keys: {list(msg['audio'].keys()) if isinstance(msg.get('audio'), dict) else type(msg.get('audio'))}")
    else:
        print(f"[{tag}] body preview:\n{json.dumps(obj, ensure_ascii=False)[:2000]}")
    # save full json
    jpath = os.path.join(OUT, f"{tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] full json -> {jpath}")
    return obj


def step1_tts():
    print("STEP1: Xiaomi TTS -> voice request")
    payload = {
        "model": "mimo-v2.5-tts",
        "messages": [
            {
                "role": "user",
                "content": "用明亮活泼的青年女声，语速适中，语气友好热情，像在和一个全能AI语音助手打招呼。",
            },
            {"role": "assistant", "content": SPEECH},
        ],
        "audio": {"format": "wav", "voice": "冰糖"},
    }
    headers = {
        "Authorization": f"Bearer {MIMO_KEY}",
        "Content-Type": "application/json",
    }
    status, body = post_json(f"{MIMO_BASE}/chat/completions", headers, payload)
    obj = summarize_response(status, body, "01_tts_request")
    if status != 200 or not obj:
        print("TTS failed, cannot proceed with voice input")
        return None
    path = save_audio_from_message(obj["choices"][0]["message"], "01_voice_request_to_omni")
    return path


def build_audio_user_content(wav_path, prompt_text, data_uri_prefix=True):
    with open(wav_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    if data_uri_prefix:
        data = f"data:audio/wav;base64,{b64}"
    else:
        data = b64
    return [
        {
            "type": "input_audio",
            "input_audio": {"data": data, "format": "wav"},
        },
        {"type": "text", "text": prompt_text},
    ]


def step2_omni_audio_out(wav_path):
    print("\nSTEP2: send voice to qwen omni, request audio/music reply")
    user_content = build_audio_user_content(
        wav_path,
        "你刚才听到的是一条语音请求。请理解内容，并用音频回复：生成/演唱一段简短欢快的中文儿歌或小旋律（约10秒）。若无法直接生成音乐音频，请用歌声/拟声哼唱并说明能力边界。",
        data_uri_prefix=True,
    )
    payload = {
        "model": OMNI_MODEL,
        "modalities": ["text", "audio"],
        "audio": {"format": "wav", "voice": "alloy"},
        "messages": [
            {
                "role": "system",
                "content": "你是全能多模态助手，支持语音输入与音频输出。",
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {OMNI_KEY}",
        "Content-Type": "application/json",
    }
    status, body = post_json(f"{OMNI_BASE}/chat/completions", headers, payload, timeout=180)
    obj = summarize_response(status, body, "02_omni_audio_modalities_datauri")
    if obj and obj.get("choices"):
        save_audio_from_message(obj["choices"][0]["message"], "02_omni_audio_reply")
    return obj


def step2b_omni_b64(wav_path):
    print("\nSTEP2b: same audio, raw base64 (no data URI)")
    user_content = build_audio_user_content(
        wav_path,
        "请理解这段语音，并用音频回复一首简短欢快的中文儿歌旋律（约10秒）。",
        data_uri_prefix=False,
    )
    payload = {
        "model": OMNI_MODEL,
        "modalities": ["text", "audio"],
        "audio": {"format": "wav", "voice": "Chelsie"},
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {OMNI_KEY}",
        "Content-Type": "application/json",
    }
    status, body = post_json(f"{OMNI_BASE}/chat/completions", headers, payload, timeout=180)
    obj = summarize_response(status, body, "03_omni_audio_raw_b64")
    if obj and obj.get("choices"):
        save_audio_from_message(obj["choices"][0]["message"], "03_omni_audio_reply_rawb64")
    return obj


def step2c_omni_text_only_music(wav_path):
    print("\nSTEP2c: voice in, ask for music - text modalities only (baseline)")
    user_content = build_audio_user_content(
        wav_path,
        "请听这段语音，复述内容，并说明你是否能直接返回音乐音频文件。",
        data_uri_prefix=True,
    )
    payload = {
        "model": OMNI_MODEL,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {OMNI_KEY}",
        "Content-Type": "application/json",
    }
    status, body = post_json(f"{OMNI_BASE}/chat/completions", headers, payload, timeout=120)
    obj = summarize_response(status, body, "04_omni_understand_voice_text")
    return obj


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"OUT={OUT}")
    tts_path = None
    try:
        tts_path = step1_tts()
    except Exception:
        traceback.print_exc()
    if not tts_path:
        print("FATAL: no voice file")
        return
    print(f"\nVoice request file ready: {tts_path} size={os.path.getsize(tts_path)}")

    try:
        step2_omni_audio_out(tts_path)
    except Exception:
        traceback.print_exc()
    try:
        step2b_omni_b64(tts_path)
    except Exception:
        traceback.print_exc()
    try:
        step2c_omni_text_only_music(tts_path)
    except Exception:
        traceback.print_exc()

    print("\n===== files =====")
    for name in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, name)
        if os.path.isfile(p):
            print(f"{name:50s} {os.path.getsize(p):10d}")


if __name__ == "__main__":
    main()
