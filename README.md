# 🎙️ MyColab — ComfyUI + Fish Audio Drama 3 雲端語音合成工作區

本專案提供在 **Google Colab (L4 / High-RAM GPU)** 上一鍵部署 **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** 與 **[PedroMarinhoDev/ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3)**（Fish Audio Drama 3 35B MoE 語音模型 + Qwen3-ASR 語音辨識）的自動化工具鏈，專門針對**數學教學影片旁白合成**與**跨語言/台灣口音混血聲音克隆**設計。

---

## 📁 專案結構說明

| 檔案 / 資料夾 | 說明 |
| :--- | :--- |
| [`ComfyUI_Drama3_Colab.ipynb`](./ComfyUI_Drama3_Colab.ipynb) | **Colab 互動式筆記本**：上傳至 Google Colab 網頁即可透過按鈕安裝、下載模型並開啟 Cloudflare Tunnel 進入 ComfyUI 網頁介面。 |
| [`setup_drama3.sh`](./setup_drama3.sh) | **CLI 自動化部署腳本**：供 `google-colab-cli` 在遠端虛擬機一鍵安裝 ComfyUI、ComfyUI-Drama3 節點與高速下載模型權重（預設 `int8`）。 |
| [`synthesize_samples.py`](./synthesize_samples.py) | **無頭（Headless）批次語音合成腳本**：不需開啟 ComfyUI 網頁，直接調用 `ComfyUI-Drama3` 底層節點進行 Qwen3-ASR 轉錄、雙語聲線混血、儲存 `.safetensors` 音色特徵檔與批次台詞合成。 |
| `outputs/` | 存放從雲端合成並下載回本機的 `.wav` 語音樣本與 `.safetensors` 專屬音色特徵檔（幾 KB 大小，可永久重複使用）。 |
| `VoiceSample/` *(已加入 `.gitignore`)* | **私人原始參考音檔目錄**：存放未公開的個人或聲優原始取樣音檔，**不會被推送至 GitHub** 以保護隱私與版權。 |

---

## 💻 跨裝置環境建置指南（在新電腦繼續工作）

當你在另一台 Windows / macOS / Linux 裝置拉取（Clone）本專案後，只需依照以下步驟即可恢復完整的 Agent / CLI 雲端控制工作流：

### 1. 取得專案並建立私人音檔目錄
```bash
git clone https://github.com/ChroHarp/MyColab.git
cd MyColab
mkdir -p VoiceSample
```
*(若需重新提取新聲線，請將乾淨的 5~15 秒參考音檔放入 `VoiceSample/`；若直接使用已提煉好的 `outputs/*.safetensors` 音色特徵檔，則無需原始音檔！)*

### 2. 安裝 `google-colab-cli`（WSL / Linux / macOS）
若在 Windows 上，請先開啟 PowerShell（系統管理員）執行 `wsl --install -d Ubuntu`，進入 Ubuntu 終端機後執行：

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
pip install --break-system-packages google-colab-cli

# 將執行檔路徑加入環境變數
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 登入 Google AI Pro 帳號並檢查計算單元餘額
colab usage
```

---

## 🚀 標準工作流程（數學教學影片配音）

### 流程一：前期測試與對軸（本機免費）
1. 在數學動畫專案（如 `ManimGX-Edu`）中，前期使用免費的 **Edge TTS (`zh-TW-YunJheNeural` 台灣男聲)** 快速生成預覽音檔並調整動畫時間軸與字幕。

### 流程二：雲端 Drama 3 高音質批次合成（定稿階段）
當台詞定稿後，透過 WSL 終端機（或直接請 Antigravity AI Agent 代勞）執行以下指令：

```bash
# 1. 啟動一台雲端 L4 GPU 虛擬機（內建 ~52GB 系統 RAM，約消耗 3.0 點/小時）
colab new --gpu L4

# 2. 上傳並執行環境與模型部署（約 2 分鐘完成 38.7GB 模型下載）
colab upload setup_drama3.sh /content/setup_drama3.sh
echo '!bash /content/setup_drama3.sh int8' | colab exec --timeout 1800

# 3. 上傳參考音檔（或已保存的 .safetensors 音色檔）與合成腳本並執行
colab upload synthesize_samples.py /content/synthesize_samples.py
echo '!python /content/synthesize_samples.py' | colab exec --timeout 1800

# 4. 下載打包好的合成結果回本機
colab download /content/drama3_samples.tar.gz ./drama3_samples.tar.gz

# 5. 【重要】立即關閉雲端虛擬機，停止扣除計算單元！
colab stop
colab usage
```

---

## 🎧 數學公式中文讀音與語氣控制技巧

在 `Fish Audio Drama 3` 中，建議將 LaTeX 數學符號轉換為**自然的中文口語讀法**，並搭配 `Direction:` 與行內標籤（如 `[pause]`、`<emphasis>...</emphasis>`）：

| 原始數學公式 | 推薦餵給 Drama 3 的台詞寫法 |
| :--- | :--- |
| 當 $a = b$ 時，$\frac{a+b}{2} = \sqrt{a \cdot b}$ | `當 a 等於 b 時，[pause] 二分之 a 加 b，等於根號 a 乘以 b。 [language: zh]` |
| 最美的公式：$e^{\pi i} = -1$ | `最美的公式：[pause] <emphasis>e 的 pi i 次方，等於負一。</emphasis> [language: zh]` |

---

## 💡 常見問題與硬體注意事項
* **為什麼不能用標準版 T4 GPU？**  
  `Drama 3` 是 35B MoE 模型，採用 `Experts-in-RAM` 技術將專家權重放在系統主記憶體（System RAM）。標準版 T4 僅有 12.7GB 系統 RAM 會直接崩潰；**L4 GPU** 預設配備 **52GB 系統 RAM**，可完美運行 `int8` 滿血版模型。
* **如何避免忘記關機扣點？**  
  每次合成任務結束後，請務必執行 `colab stop`，並用 `colab usage` 確認 `Active assignments: 0` 且 `Usage rate: 0.00/hr`。
