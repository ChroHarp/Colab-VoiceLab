# 🎙️ Colab-VoiceLab — 數學教學影片 AI 語音合成與聲線轉換實驗室

本專案旨在為數學教學動畫專案（如 `ManimGX-Edu`）打造高辨識度、自然且具台灣口音的旁白語音管線。結合 **本機 Edge TTS (`zh-TW-YunJheNeural`)** 與 **Google Colab 雲端算力**，探索並實作次世代 TTS 與零樣本音色轉換（Zero-Shot Voice Conversion）工作流。

---

## 📌 目前專案進度與交接報告 (Handover Report for Next Agent)

### 1. 已完成的基礎建設
* **WSL + `google-colab-cli` 遠端操控環境**：
  * 已於本機 WSL (Ubuntu) 安裝 `google-colab-cli (v0.7.5)` 並完成 Google AI Pro 帳號授權（目前餘額充足，約 `399.29` 計算單元）。
  * AI Agent 可直接透過 `wsl -e bash -lc "colab ..."` 指令無頭（Headless）調度雲端 GPU 虛擬機、上傳腳本、下載產出物並執行 `colab stop` 關機。
* **Phase 1 實驗（`Fish Audio Drama 3` 35B MoE 模型部署與測試）**：
  * 已完成 [`ComfyUI_Drama3_Colab.ipynb`](./ComfyUI_Drama3_Colab.ipynb)、[`setup_drama3.sh`](./setup_drama3.sh) 與 [`synthesize_samples.py`](./synthesize_samples.py)。
  * 實測在 Colab `L4 GPU`（52GB System RAM / 23GB VRAM）可在 2 分鐘內完成 38.7GB `int8` 模型下載與推論。

---

### 2. Phase 1 (`Drama 3` 直接合成) 實測問題診斷與復盤

在第一輪測試中，我們嘗試以 `VoiceSample/` 中的日本聲優（津田健次郎）綜藝節目旁白音檔，以及 `ManimGX-Edu` 快取的 Edge TTS 台灣雲哲音檔，透過 Drama 3 合成兩句數學台詞：
1. `當 a = b 時，(a+b)/2 = \sqrt{a*b}`
2. `最美的公式：e^{\pi i} = -1`

**實測產出（位於 `outputs/`）出現以下三大問題，原因已查明：**

| 測試項目 | 實測現象 | 根本原因分析 (Root Cause) |
| :--- | :--- | :--- |
| **`01_drama3_from_voicesample_*`**<br>*(純日文取樣跨語言合成)* | 產出音檔仍在講日文，像在接續原始日文取樣 | 1. **切句不完整**：用 `ffmpeg -t 10` 硬切在「ここで問題」這類未完待續的懸念句，觸發自回歸模型強烈的日文接龍慣性。<br>2. **背景環境音干擾**：電視節目片段含有微弱背景音，影響 Codec 特徵純淨度。 |
| **`02_drama3_from_edge_yunjhe_*`**<br>*(Edge TTS 雲哲轉 Drama 3)* | 中文與英文字母交界處發音語調怪異 | 1. **中英混排與拉丁拼音衝突**：台詞寫作 `e 的 pi i 次方` 並鎖定 `[language: zh]`，導致模型將 `pi`、`e`、`i` 誤判為漢語拼音或產生聲調扭曲。<br>2. **參考音檔為 +15% 加速機械音**：拿加速過的 Edge TTS 當參考，導致模型模仿出緊繃不自然的語調。 |
| **`03_drama3_blended_*`**<br>*(日文聲優 + 台灣雲哲硬拼接)* | 咬字含糊不清，甚至聽不懂在說什麼 | **聲學特徵衝突（Babbling）**：將「日本低音大叔（日文、有環境音）」與「台灣雲哲（中文、加速合成音）」用 `ffmpeg concat` 硬接為同一個 `voice0` 參考音檔，導致模型在兩種截然不同的頻率與語速間崩潰。 |

---

### 3. 🎯 下一階段核心任務：**方向 A — 二段接力法 (Edge TTS + Seed-VC / RVC 零樣本變聲)**

**使用者已確認傾向採用「方向 A（TTS + Voice Conversion 二段接力）」作為接下來的開發主軸！**

#### 為什麼「方向 A」最適合數學教學專案 (`ManimGX-Edu`)？
1. **100% 鎖定台灣口音與數學破音字**：
   第一棒繼續使用本機免費、咬字精準且已調校好破音字字典（`manimgx_tw_helper.py`）的 **Edge TTS 台灣男聲 (`zh-TW-YunJheNeural`)**。每個中文字的四聲、台灣腔調、英文數學變數（$a, b, e, \pi, i$）的讀音與停頓時間軸完全由第一棒鎖定，**絕對不會發生胡言亂語或走音**。
2. **100% 換上日本聲優的磁性聲帶（跨語言無痛變聲）**：
   第二棒使用專門的**零樣本音色轉換模型（推薦 [`Seed-VC`](https://github.com/Plachtaa/seed-vc) 或 RVC）**。這類模型不負責「唸字」，只負責把第一棒音檔的「音色（Timbre / 共鳴 / 磁性低音）」替換成目標聲優（即使目標聲優只有日文音檔也完全不影響中文咬字！）。
3. **極致輕量、超快、超省錢**：
   `Seed-VC` 模型僅約 **1~2 GB**（不到 Drama 3 的十分之一），無需 52GB 高記憶體，在免費 T4、L4 甚至本機 GPU 都能秒速批次轉換整部數學影片的旁白音軌！

#### 📋 接手 Agent 的待辦執行清單 (TODOs for Next Agent)
當使用者下令開工時，請依序執行：
1. **檢查 `VoiceSample/` 中的參考音檔**：
   * 讀取使用者準備好的乾淨聲優乾聲（或使用 Demucs / UVR5 自動去背音與切出乾淨的 10~15 秒人聲）。
2. **建立 `Seed-VC` 自動化轉換腳本 (`convert_voice_seedvc.py`)**：
   * 支援輸入單一音檔或整個 `ManimGX-Edu/<Project>/voice/*.mp3` 快取目錄。
   * 支援調整音高偏移（`pitch_shift` / `semi-tones`，例如將雲哲的音高微調對齊目標低音聲優）與音色混合比例（若想創造無版權爭議的專屬混血聲線）。
3. **在 Colab (或本機) 測試合成上述兩句數學台詞**，並將結果輸出至 `outputs/` 供使用者試聽驗收。

---

## 🎤 如何挑選更好的 Voice Sample（取樣音檔建議）

若要讓「方向 A（Seed-VC 音色轉換）」或任何 AI 語音模型發揮 100% 頂級音質，挑選參考音檔（`VoiceSample/`）時建議遵循以下 **4 大黃金準則**：

1. **無背景音樂（No BGM / 無環境雜音）—— 最重要！**
   * 綜藝節目旁白通常墊有輕快 BGM 或現場收音殘響，AI 會把背景音樂的高頻當成聲優嗓音的一部分（導致轉出來有金屬電音感）。
   * **推薦來源**：聲優的 **朗讀劇（Reading Drama）無配樂段落、廣播節目（Radio）清談開頭、有聲書（Audiobook / Audible）、遊戲角色語音包（拆包純人聲）**，或是先經過 **UVR5 (Ultimate Vocal Remover)** 去背音處理。
2. **語氣平穩、接近「講課 / 敘述」狀態**
   * 避免挑選「大吼大叫、誇張搞笑、哭腔或極度壓低氣音」的戰鬥台詞。
   * 挑選聲優用**自然、沉穩、中速**在敘述事情的片段，轉出來的數學解說聽起來最舒服耐聽。
3. **長度僅需 `10 ~ 20 秒` 即可（重質不重量）**
   * 零樣本模型不需要長篇大論，只要 10~15 秒**連續、乾淨、沒有別人插話**的純人聲，效果遠勝過 5 分鐘但含有雜音的音檔。
4. **切在完整的句子結尾**
   * 頭尾保留約 0.2 秒自然安靜，不要在聲優講話講到一半時硬切斷。

---

## 💻 跨裝置環境建置與常用指令

在新電腦拉取本專案後：
```bash
git clone https://github.com/ChroHarp/Colab-VoiceLab.git
cd Colab-VoiceLab
mkdir -p VoiceSample
```

### WSL 遠端控制 Colab 常用指令
```bash
# 查看目前點數餘額與是否有未關閉的機器
colab usage
colab status

# 開啟一台雲端 GPU 機器 (T4 或 L4)
colab new --gpu L4

# 關閉雲端機器 (務必確認 Active assignments 歸零)
colab stop
```
