import base64
import json
import os
import urllib.error
import urllib.request

OMNI_BASE = "http://llm-model-hub-proxy.sit.sf-express.com/v1"
OMNI_KEY = os.environ["OMNI_KEY"]
OUT = r"D:\butheisDelight\omni_voice_test"
wav = os.path.join(OUT, "01_voice_request_to_omni.wav")
b64 = base64.b64encode(open(wav, "rb").read()).decode()

payload = {
    "model": "aliyun/qwen3.5-omni-plus",
    "modalities": ["text", "audio"],
    "audio": {"format": "wav", "voice": "Serena"},
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/wav;base64," + b64,
                        "format": "wav",
                    },
                },
                {
                    "type": "text",
                    "text": "请理解这条语音请求，用音频唱出一段简短欢快的中文儿歌（约10秒），并附文字说明。",
                },
            ],
        }
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": True,
}

data = json.dumps(payload).encode()
req = urllib.request.Request(
    OMNI_BASE + "/chat/completions",
    data=data,
    headers={
        "Authorization": "Bearer " + OMNI_KEY,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read().decode("utf-8", "replace")
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", "replace")
    print("HTTP", e.code)
except Exception as e:
    print("EXC", e)
    body = ""

open(os.path.join(OUT, "09_stream_serena_music.txt"), "w", encoding="utf-8").write(body)
print("bytes", len(body))
print(body[:2000])
print("----- markers -----")
for key in ["audio", "data", "error", "usage", "finish", "[DONE]"]:
    print(key, body.lower().count(key.lower()))

# parse deltas
audio_b64_parts = []
text_parts = []
for line in body.splitlines():
    if not line.startswith("data:"):
        continue
    p = line[5:].strip()
    if not p or p == "[DONE]":
        continue
    try:
        obj = json.loads(p)
    except Exception:
        continue
    for ch in obj.get("choices") or []:
        delta = ch.get("delta") or {}
        if delta.get("content"):
            text_parts.append(delta["content"])
        if delta.get("audio"):
            audio = delta["audio"]
            print("DELTA AUDIO", type(audio), list(audio.keys()) if isinstance(audio, dict) else str(audio)[:80])
            if isinstance(audio, dict) and audio.get("data"):
                audio_b64_parts.append(audio["data"])
            elif isinstance(audio, str):
                audio_b64_parts.append(audio)
print("text:", "".join(text_parts)[:800])
print("audio parts", len(audio_b64_parts), "chars", sum(len(x) for x in audio_b64_parts))
if audio_b64_parts:
    raw = base64.b64decode("".join(audio_b64_parts).split(",")[-1])
    p = os.path.join(OUT, "09_omni_serena_music.wav")
    open(p, "wb").write(raw)
    print("saved", p, len(raw))
