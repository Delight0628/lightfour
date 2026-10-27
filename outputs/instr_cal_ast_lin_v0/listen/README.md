# 试听包 · INSTRUMENT_CALIBRATION · ast_lin_v0

生成时间：2026-10-27   条目数：497

- `probe_in_vocab/`：34 条 in-vocab 参考音频（文件名前缀 = 真值 sound_id）
- `probe_out_of_vocab/`：61 条 out_of_vocab 参考音频（**不计分**，只供人工听感核对）
- `perturb_in_vocab/`：in-vocab 34 条 ref 的扰动样本（P5 缺项，见卡片 A5）
- `failures/`：28 条失败案例（文件名 = `truth<真值>_pred<预测>_<ref_id>.mp3`）
  **不含 sid 25 `K\``**：该类训练样本 0，标注「该类不可学」，按口径不计入失败案例。

口径提醒：本试听包只服务**仪器校准**；音频均为 seedlib 参考素材与本地扰动产物，不外传、不入 git。
