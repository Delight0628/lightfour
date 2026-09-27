import base64
import json
import os
import time
import urllib.error
import urllib.request

OMNI_BASE = "http://llm-model-hub-proxy.sit.sf-express.com/v1"
OMNI_KEY = os.environ["OMNI_KEY"]
OUT = r"D:\butheisDelight\omni_voice_test"


def post(payload, timeout=90):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OMNI_BASE + "/chat/completions",
        data=data,
        headers={
            "Authorization": "Bearer " + OMNI_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body.decode())
        except Exception:
            return e.code, {"raw": body[:400].decode("utf-8", "replace")}
    except Exception as e:
        return -1, {"exc": str(e)}


def probe_voice(v):
    payload = {
        "model": "aliyun/qwen3.5-omni-plus",
        "modalities": ["text", "audio"],
        "audio": {"format": "wav", "voice": v},
        "messages": [{"role": "user", "content": "用语音说：测试。"}],
        "max_tokens": 256,
        "temperature": 0.2,
    }
    return post(payload)


def try_save_audio(name, obj):
    if not obj or "choices" not in obj:
        return False
    msg = obj["choices"][0].get("message", {})
    audio = msg.get("audio")
    print("  message keys:", list(msg.keys()))
    print("  content:", str(msg.get("content"))[:300])
    print("  usage:", obj.get("usage"))
    if not audio:
        print("  NO audio field")
        return False
    if isinstance(audio, dict):
        print("  audio keys:", list(audio.keys()), "meta:", {k: audio[k] for k in audio if k != "data"})
        data = audio.get("data")
    else:
        data = audio
        print("  audio is", type(audio), "len", len(str(audio)))
    if not data:
        return False
    raw = base64.b64decode(str(data).split(",")[-1])
    path = os.path.join(OUT, name + ".wav")
    open(path, "wb").write(raw)
    print("  SAVED", path, len(raw))
    return True


def main():
    voices = [
        "Serena",
        "Ethan",
        "Chelsie",
        "Aimee",
        "Denny",
        "mimo_default",
        "冰糖",
        "Cherry",
        "qwen-tts",
        "Sambert",
        "longxiaochun",
        "longwan",
        "loongstella",
        "loongbella",
        "zhimiao-emo",
        "female",
        "male",
        "default",
        "Chinese",
        "Xiaoxiao",
        "Xiaoyi",
        "Yunxi",
        "Yunyang",
        "ruoxi",
        "sisi",
        "zhichu",
        "Aiden",
        "Roger",
        "Katerina",
        "",
    ]
    for v in voices:
        print(f"\n==== probe voice={v!r} ====")
        code, obj = probe_voice(v)
        print("  status", code)
        if code != 200:
            err = json.dumps(obj, ensure_ascii=False)
            print(" ", err[:350])
            # keep going even on 429; stop only if auth broken
            if "unauthorized" in err.lower() or "invalid api key" in err.lower():
                break
            time.sleep(5)
            continue
        try_save_audio(f"07_omni_speech_{(v or 'empty').replace('/', '_')}", obj)
        open(os.path.join(OUT, "omni_voice_success.json"), "w", encoding="utf-8").write(
            json.dumps(obj, ensure_ascii=False, indent=2)[:300000]
        )
        print("SUCCESS voice=", v)
        return v
    print("\nNo successful voice yet. Trying stream without voice field...")
    payload = {
        "model": "aliyun/qwen3.5-omni-plus",
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "messages": [{"role": "user", "content": "请用语音说：你好，语音通道测试。"}],
        "max_tokens": 256,
        "temperature": 0.2,
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
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode("utf-8", "replace")
            open(os.path.join(OUT, "08_stream_novoice.txt"), "w", encoding="utf-8").write(body)
            print("stream body first 1500:\n", body[:1500])
            print("stream has audio?", "audio" in body.lower())
    except urllib.error.HTTPError as e:
        print("stream HTTP", e.code, e.read()[:500])
    except Exception as e:
        print("stream EXC", e)
    return None


if __name__ == "__main__":
    main()
