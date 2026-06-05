# AI Tuner — 项目分析与备忘

> 本文档供 AI 编程助手（Claude Code / Reasonix）混用时参考。
> 记录当前项目的技术决策风险点、已知问题和行动建议。

---

## ⚠️ 高风险区域

### 1. 训练收敛风险（HuBERT 路径未经验证）

旧版 Mel→Mel 架构训了几个 epoch 后 loss 不再变化——模型学会了恒等映射偷懒。新版改用 HuBERT 换输入空间来解决这个问题，**但还没训练验证过**。

**第一个 epoch 重点看 `loss_mel`**：
- 如果 `loss_mel` < 0.3：model 又在偷懒，需要调整（增大数据偏移量、减少 `min_clean_ratio`、或加噪声）
- 如果 `loss_mel` > 0.5 且逐步下降：路径正确

### 2. `hifi_gan.py` 与 `neural_vocoder.py` 架构同步

两个文件有重复的 ResBlock / MRF / _make_pitch_feature 定义，且训练端用 `weight_norm`，推理端不需要。**每次改 HiFiGANGenerator 的结构都要同步更新 InferenceGenerator**。

当前（2026-06-05）已确认 InferenceGenerator 修复完毕，架构与 HiFiGANGenerator 一致。以后改结构时注意：
- `hifi_gan.py`：HiFiGANGenerator（训练用，带 weight_norm）
- `neural_vocoder.py`：InferenceGenerator（推理用，无 weight_norm）
- 训练导出：`generator.remove_weight_norm()` → 存 `state_dict` → 推理端 `load_state_dict` 加载
- 如果加了新层或改名，两边要同步

### 3. 单歌手模型泛化失败（已知）

`models/tuner_female_only.pth`（OpenCpop 单女声训练）已废弃——男声会出电音。多歌手训练数据（M4Singer / OpenSinger）是必须的，不是可选。

---

## 🔶 值得商榷的技术选型

### 1. HuBERT 模型：英文 vs 中文

| 当前 | 建议 |
|------|------|
| `facebook/hubert-base-ls960` | `TencentGameMate/chinese-hubert-base` |

当前用的 HuBERT 是英文 ASR 预训练的（LibriSpeech 960h），对中文歌声的音素建模可能不理想。但换模型要 **benchmark 验证**——跑相同测试集，对比两个模型的 `loss_mel` 和主观音质。不是无脑换。

### 2. HiFi-GAN vs DDSP vocoder（大偏移量音质）

HiFi-GAN 是纯 CNN 生成谐波——谐波是 CNN 近似出来的，音高改动越大失真越严重。DDSP 用加法合成器生成谐波（数学上精确的正弦波叠加），只改 F0 输入就能生成正确的谐波。

**判断标准**：训完 HiFi-GAN 后，用 > 300 cents 的偏移量测试。如果电音/机械感明显 → 考虑换 DDSP 解码器。前端 HuBERT 不变，只换 vocoder 层。

### 3. pyrubberband 版本锁死

`backend/requirements.txt` 锁了 `pyrubberband==0.3.0`（只支持标量 pitch_shift），导致 `tuner.py` 里需要手动分段 + crossfade（`_apply_per_frame_pitch_shift`，约 120 行代码）。

pyrubberband 0.4+ 已支持数组 `n_steps`，可以大幅简化 DSP 管线。但升级前要 **AB 测试**对比音质——旧版的 per-frame 逻辑经过打磨，新版可能有不同的 artifact。

### 4. DSP/AI 混合模式未实现

README 里设计了 `< 2 半音走 DSP，≥ 2 半音走 AI` 的混合策略，但代码里没有。`tuner.py` 的三个管线（scale / reference / neural）是独立的三条路，没有按帧粒度混合的版本。

实现时需要处理：
- 按帧判断偏移量 → 路由到 DSP 或 AI
- 两段之间 crossfade 拼接（避免边界硬断）
- AI 管线需要先跑整个音频的 HuBERT 特征（一次），再按需调用 vocoder

### 5. Overlap-add 推理未实现

训练用 800ms 片段，推理时传入 3 分钟完整歌曲 → 分布外。README 描述了对策（分短段 vocoder + Hanning 窗 crossfade 拼接），但 `tuner.py:641` 的 neural 推理是把整个音频一次喂入 vocoder，没有 overlap-add 逻辑。

训完模型后需要实现，否则长音频推理可能有帧边界相位跳变。

---

## 🟡 工程债务（非紧急但应关注）

### 1. 并发推理无保护

`backend/app.py` 的 neural 端点没有请求队列——多用户同时调用会撑爆 GPU 显存。如果只打算本地单用户用，暂时不影响。

### 2. 缺少人声/伴奏分离

只能修干声。用户上传带伴奏的音频会出问题。README 写了 TODO（demucs），优先级低于训练。

### 3. `tuner.py` `_apply_reverb` 用固定 seed

`np.random.default_rng(seed=42)` 硬编码——每次生成的 IR 完全一样。功能上没问题，但如果后续要增加随机性避免过拟合检测，需要去掉固定 seed。

---

## 📋 当前状态

| 模块 | 状态 |
|------|------|
| DSP 修音（scale + reference） | ✅ 完成 |
| 节奏修正（grid + reference） | ✅ 完成 |
| 后处理（pitch smooth + 响度归一 + 微混响） | ✅ 完成 |
| Web UI（4 tab） | ✅ 完成 |
| 训练数据生成（WORLD + HuBERT） | 🔄 生成中（云机器） |
| 模型训练 | ⬜ 待执行 |
| 推理模型（InferenceGenerator） | ✅ 刚修完 bug |
| Overlap-add 推理 | ⬜ 训完模型后实现 |
| DSP/AI 混合模式 | ⬜ 未实现 |
| 人声分离预处理 | ⬜ 未实现 |

---

## 🔧 快速启动命令

```bash
# 仅 DSP 推理（不需要模型）
pip install -r backend/requirements.txt
cd backend && uvicorn app:app --host 0.0.0.0 --port 8050 --reload

# 训练（云机器）
pip install -r scripts/requirements_train.txt
python scripts/generate_training_data.py --input_dir data/clean/ --output_dir data/training/ --use_hubert
python scripts/train.py --data_dir data/training/ --checkpoint_dir checkpoints/ --batch_size 16 --num_epochs 100

# 训练完成后，tuner.pth 放到 models/ 目录即可启用 neural 端点
```
