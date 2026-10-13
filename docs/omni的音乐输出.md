# omni的音乐输出

**我：** AGENT_DEFAULT_LLM_PROVIDER=openai
AGENT_DEFAULT_LLM_MODEL=aliyun/qwen3.5-omni-plus
AGENT_DEFAULT_LLM_BASE_URL=http://llm-model-hub-proxy.sit.sf-express.com/v1
AGENT_DEFAULT_LLM_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJhcGkyXzRmNGZiMGZlLTE2NzMtNDNjOS1iOTFlLTdmYTJkNGNmYzhhMyIsImVudiI6InNpdCIsImp0aSI6MzMwODksInByb2plY3RfaWQiOjI4NDcxLCJzeXN0ZW1LZXkiOiIzOTRhZmU3Ny1kODFmLTQ0ZTktOTM2Zi1jMzA2ZWQyYWYzZDUifQ.7d7XWujq-uxCTuz053pdMduviPSnTOe1qoNLFTRA3j8
AGENT_DEFAULT_LLM_TEMPERATURE=0.0
AGENT_DEFAULT_LLM_MAX_TOKENS=32000

调一下如上模型看看是否联通

**Xiaomi MiMo:** 模型端点已联通，测试通过。

| 项 | 结果 |
|---|---|
| Base URL | `http://llm-model-hub-proxy.sit.sf-express.com/v1` |
| 模型 | `aliyun/qwen3.5-omni-plus`（实际返回模型名 `qwen3.5-omni-plus`） |
| 认证 | Bearer JWT 正常 |
| 响应 | HTTP 200，`chat.completion` |
| 示例回复 | `**Pong!** ...` |
| 用量 | prompt 17 / completion 32 / total 49 |

请求走了 OpenAI 兼容的 `/chat/completions`，温度 0、`max_tokens=32` 下正常出字，鉴权与路由都可用。

**我：** 这个aliyun/qwen3.5-omni-plus模型是全能的模型，帮我测试他的语音回复能力，比如说让他生成一段音乐回复过来，你也用语音发过去

语音合成用小米的模型文档是https://mimo.mi.com/docs/zh-CN/quick-start/summary/welcome，key是sk-cvfce7itjivfie5htqje4j2wn8x1z52ct3b6k5oxrwwxqvhx

**Xiaomi MiMo:** ## 测试结论

`aliyun/qwen3.5-omni-plus` **语音输入可用，语音输出在流式下可用**；但它**不会生成音乐音频**，只会把文字回答合成成语音。

### 已跑通的链路

| 步骤 | 结果 | 文件 |
|---|---|---|
| 小米 TTS 合成语音请求（冰糖，13.9s） | 成功 | `01_voice_request_to_omni.wav` |
| 把这段语音发给 omni（`input_audio`） | 成功，`prompt_tokens` 含 `audio_tokens: 93`，模型复述了请求 | — |
| omni **流式**语音回复（voice=`Serena`） | 成功，约 71s PCM→WAV | `09_omni_serena_speech.wav` |
| 小米 TTS **唱歌模式**生成音乐（`(唱歌)`，10.9s） | 成功 | `05_xiaomi_tts_music.wav` |

### 关键发现

1. **非流式拿不到音频**：即使带 `modalities:["text","audio"]`，响应里只有 `content`/`reasoning_content`，没有 `message.audio`；但 `usage.completion_tokens_details.audio_tokens` 有值，说明后端算了音频 token。
2. **流式才能取到音频**：`stream:true` + `audio.voice=Serena` 时，SSE 的 `delta.audio.data` 会连续吐 base64 PCM。`alloy`/`Cherry` 会被拒：`Voice 'xxx' is not supported`。
3. **音乐能力边界**：omni 明确说自己「无法直接生成或发送音频文件（包括唱歌）」，语音回复内容是**儿歌歌词 + 旋律建议（do re mi）**，并建议用小米 TTS 做成音频。真正的音乐音频是用小米 TTS 的 `(唱歌)` 标签生成的。
4. **网关限流较紧**：项目 28471 对这个模型多次 429，重试需间隔。

### 可用参数备忘

- Omni 代理：`http://llm-model-hub-proxy.sit.sf-express.com/v1`
- 音频输出：`modalities:["text","audio"]` + `audio:{"format":"wav","voice":"Serena"}` + **必须 stream**
- 小米 TTS：`https://api.xiaomimimo.com/v1`，`mimo-v2.5-tts`，音色 `冰糖`，目标文本放 `assistant` 消息

三段音频已在预览面板可播：先听发出去的语音，再听 omni 的语音回复，最后是小米 TTS 唱的儿歌。