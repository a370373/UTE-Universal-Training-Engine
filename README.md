## 🚀 UTE — Universal Training Engine

通用訓練引擎（Universal Training Engine）

Version: 2.4.0

一套以 Python 建構、基於 Hugging Face 生態系的可重用模型訓練引擎，專注於 Causal Language Model（因果語言模型）的 Supervised Fine-Tuning（SFT）、LoRA 與 QLoRA 微調。

UTE 將模型相容性檢查、硬體偵測、LoRA 注入檢查、模型量化設定、模型載入、訓練流程測試、Trainer 整合測試、Embedding 訓練，以及 Adapter 輸出驗證整合為模組化的訓練工作流程。

## 🎯 設計目標

## 讓不同模型、不同專案與不同執行環境，都能在一致的訓練流程下使用同一套引擎。

## 讓每個人都能訓練 AI Model，小模型也許能超越大模型 who knows 🤷‍♂️

---

# 📌 專案簡介

UTE（Universal Training Engine）是一個獨立、可重用的模型訓練引擎。

它並不是為某個特定模型或單一 AI 專案打造，而是將常見的 SFT + LoRA / QLoRA 訓練流程抽象成可重用的程式模組。

UTE 提供三種主要訓練配置：

- LoRA： 一般 LoRA 微調，預設訓練模式。
- QLoRA 4-bit： 透過 4-bit 量化載入 Base Model，搭配 LoRA 進行參數高效微調。
- LoRA + Trainable Embeddings： 選用式訓練 Input Embedding 與 Output LM Head，與 LoRA 共用相同訓練流程。

UTE 的設計不綁定：

- 特定模型名稱
- 特定 Hugging Face Model ID
- 特定本機模型路徑
- 特定訓練資料集
- 特定 AI 專案
- 特定輸出目錄
- 單一硬體設備

只要模型與執行環境符合目前支援的技術範圍，就可以透過 UTE 執行訓練工作流程。

---

# ✨ 核心功能

# 🖥️ 1. Hardware Profiler

硬體偵測與資源資訊分析。

目前包含：

- CPU 型號與核心資訊
- CPU Logical / Physical Cores
- CPU Affinity
- CPU Quota
- 系統記憶體資訊
- Cgroup Memory 限制偵測
- NVIDIA / CUDA 環境偵測
- 建議 CPU 執行緒數量
- 硬體資訊 JSON 報告

# 🔍 2. Compatibility Checker

模型相容性檢查。

使用 Hugging Face Transformers 檢查模型相關資訊，包括：

- 模型架構辨識
- Causal Language Model 架構檢查
- Tokenizer 載入檢查
- Chat Template 檢查
- Vocabulary 相容性檢查
- 模型與 Tokenizer 相關資訊

目標是在正式訓練前，盡可能識別模型與 Tokenizer 的相容性問題。

# 🧩 3. LoRA Inspector

LoRA 注入與模型結構檢查。

包含：

- 模型載入
- Embedding 結構檢查
- Linear Modules 探測
- LoRA Target Modules 檢查
- PEFT LoRA 注入
- Trainable Parameters 統計
- Trainable Parameters Ratio 計算
- 可選式 Embedding Training 設定整合

# 🧠 4. Embedding & LM Head Training

選用式 Embedding 與 Language Model Head 訓練功能。

UTE 新增 "--train-embeddings"，允許使用者在原本的 LoRA 訓練過程中，同時訓練並保存模型的 Input Embedding 與 Output LM Head 模組。

目前包含：

- Input Embedding 動態解析
- Output LM Head 動態解析
- Embedding Module Identity 檢查
- Tied Embeddings 偵測
- PEFT "modules_to_save" 整合
- "ensure_weight_tying" 設定
- Embedding 與 LM Head 額外訓練及保存

Architecture-Generic Module Resolution

UTE 不直接假設模型使用特定模組名稱。

而是透過 Transformers 提供的模型 API：

- "get_input_embeddings()"
- "get_output_embeddings()"
- "named_modules()"

動態解析模型的 Input Embedding 與 Output LM Head 模組。

不需要將模組名稱硬編碼成：

- "embed_tokens"
- "lm_head"
- "wte"
- 其他特定模型架構的名稱

實際解析結果取決於模型提供的 API 與模組結構。如果模型無法提供或解析相關模組，UTE 會回報錯誤，而不是任意猜測模組名稱。

Tied Embeddings

部分 Causal Language Model 會讓 Input Embedding 與 Output LM Head 共用權重。

UTE 會檢查兩者是否共享權重，並依照模型設定處理 PEFT 的 Weight Tying。

實際需要保存的模組數量會依照模型架構與共享權重方式而異。

使用方式

一般 LoRA：

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output

啟用 Embedding Training：

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output \
  --train-embeddings

設計原則

- 預設停用，維持既有 LoRA 訓練行為。
- 使用相同訓練資料。
- 使用相同 Trainer 與 Loss 流程。
- 透過 PEFT "modules_to_save" 管理額外訓練模組。
- 不要求使用者手動指定 Embedding 模組名稱。
- 與原本的 LoRA Target Modules 配置共存。

此功能不等同於完整 Full Fine-Tuning。未納入 LoRA Target Modules 的 Transformer 主體權重仍然保持凍結。

# 🛡️ 5. Preflight Engine

正式訓練前的預檢引擎。

整合 Compatibility Checker 與 LoRA Inspector，協助確認：

- 模型是否符合支援範圍
- Tokenizer 是否能正常載入
- 模型結構是否符合預期
- LoRA 是否能正常注入
- 可訓練參數是否正確建立
- 選用式 Embedding Training 設定是否能正常解析

預檢結果可以輸出為 JSON 報告。

# 🧪 6. Training Smoke Test

訓練基本流程測試。

透過小型測試流程驗證：

1. Tokenization
2. LoRA Injection
3. Forward Pass
4. Backward Pass
5. Gradient 檢查
6. Optimizer Step

當啟用 "--train-embeddings" 時，也會將 Embedding Training 設定納入測試流程。

用於確認模型是否具備基本的訓練執行能力。

# 🔗 7. Trainer Integration Test

Hugging Face TRL Trainer 整合測試。

使用合成測試資料執行實際 Trainer 訓練流程，檢查：

- Dataset 建立
- SFTConfig 初始化
- SFTTrainer 初始化
- Trainer Training Step
- Adapter 儲存
- Adapter 設定檢查
- Adapter 權重檔案檢查
- Adapter 重新載入
- Embedding Training 設定保存檢查

此模組用於確認 UTE 與 TRL Trainer 的整合流程能否正常執行。

# 📦 8. Adapter Output Verifier

LoRA Adapter 輸出驗證器。

正式訓練完成後，獨立驗證輸出的 Adapter。

目前檢查項目包含：

- Adapter Configuration
- Safetensors 權重檔案
- LoRA Metadata
- Adapter 載入
- Base Model 與 Adapter 整合
- Forward Pass
- Logits 有限數值檢查
- JSON 驗證報告

不只確認檔案是否存在，也會實際重新載入 Adapter 並執行模型推論檢查。

# ⚙️ 9. Quantization Engine

模型量化設定引擎。

目前支援：

- No Quantization：一般 LoRA 模式
- 4-bit Quantization：QLoRA 模式
- NF4 Quantization
- FP4 Quantization
- Double Quantization
- BF16 Compute Dtype
- FP16 Compute Dtype
- Automatic Compute Dtype Selection
- CUDA 環境檢查
- 選用量化依賴檢查

Quantization Engine 負責解析使用者指定的量化設定，並在模型載入前確認執行環境是否符合目前 QLoRA 的需求。

# 🧠 10. Shared Model Loader

共用模型載入模組。

統一管理 Base Model 載入流程，包括：

- Hugging Face AutoModelForCausalLM
- Local Model Loading
- Model Precision Configuration
- Quantization Configuration
- CUDA Device Mapping
- 4-bit Model Loading
- QLoRA K-bit Training Preparation

一般 LoRA 與 QLoRA 共用同一套模型載入架構，避免不同訓練模式各自維護重複的載入邏輯。

---

## 🏗️ 系統架構

UTE 採用模組化設計，各個功能可以由訓練主程式統一協調。
```
                    UTE
          Universal Training Engine
                     │
                     ▼
             ┌───────────────┐
             │    train.py   │
             │  Training CLI │
             └───────┬───────┘
                     │
       ┌─────────────┼──────────────┐
       │             │              │
       ▼             ▼              ▼
  Hardware     Compatibility    Quantization
  Profiler       Checker          Engine
       │             │              │
       │             ▼              │
       │       LoRA Inspector       │
       │             │              │
       │             ▼              │
       │     Embedding Training    │
       │             │              │
       └─────────────┼──────────────┘
                     │
                     ▼
             Shared Model Loader
                     │
                     ▼
             Training Workflow
                     │
       ┌─────────────┼──────────────┐
       │             │              │
       ▼             ▼              ▼
    Smoke         Trainer       SFT Training
    Test         Integration          │
                    Test              ▼
                              Adapter Saving
                                     │
                                     ▼
                              Adapter Verifier
                                     │
                                     ▼
                                JSON Reports
```
一般 LoRA 與 QLoRA 共用訓練工作流程。

兩者主要差異在於 Base Model 的載入與量化設定，而不是各自維護一套獨立 Trainer。

Embedding Training 則是可以選擇啟用的額外可訓練模組設定。

---

# 📁 專案目錄
```
UTE-Universal-Training-Engine/
│
├── train.py
├── requirements.txt
├── requirements-qlora.txt
├── .gitignore
├── LICENSE
├── README.md
│
└── engine/
    ├── __init__.py
    ├── version.py
    ├── hardware.py
    ├── compatibility.py
    ├── quantization.py
    ├── model_loader.py
    ├── embedding_training.py
    ├── lora_inspector.py
    ├── preflight.py
    ├── smoke_test.py
    ├── trainer_integration_test.py
    └── adapter_verifier.py
```
---

# 🧰 技術架構

UTE 使用成熟的 Python AI 訓練生態系。

技術| 用途
Python| 主要開發語言
PyTorch| 深度學習與模型訓練
Transformers| 模型與 Tokenizer 管理
Datasets| 訓練資料處理
PEFT| Parameter-Efficient Fine-Tuning
TRL| Supervised Fine-Tuning Trainer
Safetensors| 模型權重安全序列化
bitsandbytes| 4-bit 模型量化支援
Accelerate| 模型載入與裝置管理支援

# 🔧 核心技術方向

- Causal Language Modeling
- Supervised Fine-Tuning
- Low-Rank Adaptation
- Quantized Low-Rank Adaptation
- Parameter-Efficient Fine-Tuning
- Trainable Embeddings
- Trainable Language Model Head
- Tied Embeddings
- 4-bit Model Quantization
- NF4 / FP4 Quantization
- Double Quantization
- Model Compatibility Inspection
- Shared Model Loading
- Training Workflow Validation
- Adapter Verification

---

# 📥 安裝方式

1. 取得專案

git clone https://github.com/a370373/UTE-Universal-Training-Engine.git

cd UTE-Universal-Training-Engine

2. 建立 Python 虛擬環境（建議）

Linux / macOS：

python3 -m venv .venv

source .venv/bin/activate

Windows：

python -m venv .venv

.venv\Scripts\activate

3. 安裝 PyTorch

請依照自己的作業系統、硬體及加速環境，選擇適合的 PyTorch 安裝方式：

https://pytorch.org/get-started/locally/

4. 安裝 UTE 依賴

pip install -r requirements.txt

此步驟安裝一般 LoRA 訓練所需的主要依賴。

5. QLoRA 4-bit 額外依賴（選用）

一般 LoRA 使用者不需要安裝額外的量化依賴。

若要使用 QLoRA 4-bit，請先確認系統具有符合需求的 NVIDIA CUDA GPU 與 PyTorch CUDA 環境，再安裝：

pip install -r requirements-qlora.txt

此檔案包含：

- bitsandbytes
- accelerate

注意：

- QLoRA 需要符合相依套件、CUDA 與硬體相容性要求。
- 目前 QLoRA 實作要求受支援的 CUDA GPU。
- 一般 LoRA 不需要安裝 bitsandbytes 與 accelerate 這組額外量化依賴。

6. 確認安裝

python train.py --help

如果正常顯示 CLI 使用說明，代表主程式可以啟動。

---

# 🚀 使用方式

UTE 使用命令列介面管理訓練流程。

一般 LoRA 訓練

一般 LoRA 是預設訓練模式。

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output

也可以明確指定：

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output \
  --quantization none

QLoRA 4-bit 訓練

透過 4-bit 量化載入 Base Model，搭配 LoRA 進行訓練。

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output \
  --quantization 4bit

QLoRA 自訂量化設定

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output \
  --quantization 4bit \
  --quant-type nf4 \
  --double-quant \
  --compute-dtype auto

Embedding & LM Head Training

啟用 Input Embedding 與 Output LM Head 額外訓練：

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output \
  --train-embeddings

此功能與原本 LoRA 訓練使用相同資料、Trainer 與 Loss 流程。

量化設定參數

參數| 說明
"--quantization"| 量化模式："none" 或 "4bit"
"--quant-type"| 4-bit 量化類型："nf4" 或 "fp4"
"--double-quant"| 啟用 Double Quantization
"--no-double-quant"| 停用 Double Quantization
"--compute-dtype"| 計算精度："auto"、"bf16" 或 "fp16"

Compute Dtype 說明

- "auto"：由引擎根據目前 CUDA GPU 的能力選擇適合的計算精度。
- "bf16"：使用 BF16 計算精度，需要硬體支援。
- "fp16"：使用 FP16 計算精度。

基本參數說明

參數| 說明
"--model"| 模型本機路徑或支援的 Hugging Face Model ID
"--data-dir"| 訓練資料目錄
"--output-dir"| Adapter 輸出目錄

常用訓練參數

參數| 說明
"--epochs"| 訓練 Epoch 數量
"--learning-rate"| Learning Rate
"--max-length"| 最大 Token Sequence Length
"--batch-size"| Batch Size
"--gradient-accumulation"| Gradient Accumulation Steps
"--lora-r"| LoRA Rank
"--lora-alpha"| LoRA Alpha
"--lora-dropout"| LoRA Dropout
"--target-modules"| LoRA Target Modules
"--train-embeddings"| 額外訓練並保存 Input Embedding 與 Output LM Head
"--cpu-threads"| CPU 執行緒設定
"--seed"| 隨機種子
"--resume-checkpoint"| 從既有 Checkpoint 恢復訓練

實際可用參數、預設值與完整使用方式，請以目前版本的 CLI 說明為準：

python train.py --help

---

# 🔎 驗證與測試模式

UTE 提供多種訓練前與訓練整合測試功能：

- Validate Only：僅執行驗證流程。
- Preflight：執行訓練前檢查。
- Smoke Test：測試基本 Forward / Backward 與 Optimizer 流程。
- Trainer Integration Test：測試 TRL Trainer 整合。

Validate Only

python train.py \
  --model /path/to/model \
  --data-dir /path/to/dataset \
  --output-dir ./output \
  --validate-only

Preflight

python train.py \
  --model /path/to/model \
  --output-dir ./output \
  --preflight

啟用 Embedding Training 的 Preflight：

python train.py \
  --model /path/to/model \
  --output-dir ./output \
  --preflight \
  --train-embeddings

Smoke Test

python train.py \
  --model /path/to/model \
  --output-dir ./output \
  --smoke-test

啟用 Embedding Training 的 Smoke Test：

python train.py \
  --model /path/to/model \
  --output-dir ./output \
  --smoke-test \
  --train-embeddings

Trainer Integration Test

python train.py \
  --model /path/to/model \
  --output-dir ./output \
  --trainer-integration-test

啟用 Embedding Training 的 Trainer Integration Test：

python train.py \
  --model /path/to/model \
  --output-dir ./output \
  --trainer-integration-test \
  --train-embeddings

各測試功能與報告輸出參數可透過：

python train.py --help

查詢。

---

# 📊 訓練資料與 Loss 設定

UTE 目前採用整段訓練序列參與 Loss 計算的設定。

目前正式訓練配置：

assistant_only_loss = False
completion_only_loss = False

也就是不只對 Assistant 回覆部分計算 Loss，而是依照目前的資料與標籤處理流程，讓整段序列參與訓練。

資料集的具體欄位、格式與載入規則，應以目前版本 "train.py" 的資料載入實作為準；不要假設所有 JSON、JSONL 或其他資料集格式都能直接使用。

---

# 🧪 開發與測試狀態

目前 UTE 已建立以下測試模組：

測試項目| 功能
Compatibility Checker| 模型相容性檢查
LoRA Inspector| LoRA 注入檢查
Preflight Engine| 訓練前檢查
Smoke Test| Forward / Backward / Optimizer 測試
Trainer Integration Test| TRL Trainer 整合測試
Adapter Output Verifier| Adapter 輸出與重新載入驗證
Quantization Engine| 量化設定與環境檢查
Shared Model Loader| 共用模型載入流程
Embedding Training| Input Embedding / Output LM Head 額外訓練

已完成的開發測試

目前開發過程中已完成：

- Compatibility Checker 測試
- LoRA Inspector 測試
- Preflight Engine 測試
- Training Smoke Test
- Trainer Integration Test
- Adapter Verifier 語法與匯入檢查
- CLI 啟動檢查
- Python Compileall 檢查
- QLoRA CLI 參數檢查
- QLoRA CPU 環境拒絕測試
- 一般 LoRA 回歸測試

Embedding Training 測試

目前已使用 Qwen3 模型完成 "--train-embeddings" 整合測試。

測試項目| 結果
動態解析 Input Embedding| PASS
動態解析 Output LM Head| PASS
Tied Embeddings 偵測| PASS
PEFT 設定整合| PASS
Preflight| PASS
Smoke Test| PASS
Trainer Integration Test| PASS
Adapter 設定保存檢查| PASS
Adapter 重新載入| PASS

測試中確認：

- Input Embedding："model.embed_tokens"
- Output LM Head："lm_head"
- Tied Embeddings："true"
- PEFT "modules_to_save"："model.embed_tokens"

上述名稱是目前 Qwen3 測試模型的實際解析結果，不是 UTE 寫死的模組名稱。

Trainer Integration Test 已確認 Adapter 設定包含預期的 "modules_to_save"，並成功完成 Adapter 保存及重新載入流程。

注意：目前的測試結果不代表所有模型架構都已完成相容性驗證，也不代表所有 Adapter 權重內容都已經過獨立逐項檢查。

一般 LoRA Smoke Test

一般 LoRA Smoke Test 包含：

- Tokenization
- LoRA Injection
- Forward Pass
- Backward Pass
- Gradient 檢查
- Optimizer Step

一般 LoRA Trainer Integration Test

已完成：

- Dataset 建立
- SFTTrainer 初始化
- Training Step
- Adapter 儲存
- Adapter 重新載入

QLoRA 測試限制

目前已完成 QLoRA 4-bit 的程式整合、CLI 驗證與 CPU 環境拒絕測試。

由於目前開發環境沒有 CUDA GPU，尚未完成實際 4-bit 模型載入與 CUDA 訓練的端到端測試。

因此，目前不能宣稱所有 CUDA GPU、模型架構及 bitsandbytes 版本組合均已通過驗證。

注意：測試模組存在，不代表所有模型、作業系統、硬體後端與依賴版本組合都已通過測試。正式訓練輸出的 Adapter 仍應透過 Adapter Verifier 實際驗證。

---

# 🧠 研究方向：High-Rank LoRA + Trainable Embeddings

UTE 新增 Embedding Training 後，也具備了進一步研究高 Rank LoRA 與額外可訓練模組的基礎。

其中一個可能的研究方向，是探索極小型 Causal Language Model 是否能透過高 Rank LoRA、可訓練 Embeddings 與大量訓練資料，取得更大的可訓練參數空間。

研究構想

以約 10M Parameters 的 Tiny Base Model 作為實驗對象，研究：

- 高 Rank LoRA
- Trainable Input Embeddings
- Trainable Output LM Head
- 不同 LoRA Target Modules
- 不同 Rank 與 Alpha 配置
- 長期持續訓練
- 大規模訓練資料
- 不同模型初始化方式

核心假設

LoRA 透過低秩矩陣表示權重更新：

[
\Delta W = BA
]

其中 Rank 會影響 LoRA 可表達的更新空間。

提高 Rank 可以增加更新矩陣的表達能力；搭配 Embedding Training，則能讓 Token Embedding 相關參數也參與訓練。

然而，高 Rank LoRA 並不自動等同於 Full Fine-Tuning。

如果 Transformer 主體中仍有未納入 LoRA Target Modules 的權重，這些權重仍然會保持凍結。

預計研究比較

實驗組| 訓練配置
A| 一般 LoRA
B| 高 Rank LoRA
C| LoRA + Trainable Embeddings
D| 高 Rank LoRA + Trainable Embeddings
E| Full Fine-Tuning

透過相同或可比較的模型架構、資料集及訓練條件，觀察：

- Training Loss
- Validation Loss
- Trainable Parameters
- 記憶體使用量
- 訓練時間
- 模型生成品質
- 不同 Rank 的表現差異

研究定位

這是尚待實驗驗證的研究構想，不代表目前 UTE 已經具備 Tiny Model Pretraining 或 Full Fine-Tuning 功能。

研究目標是探索：

能否透過高 Rank LoRA 與可訓練 Embeddings，在極小模型上建立更有效率的語言學習策略？

---

# 🖥️ 執行環境與硬體

UTE 的設計目標是避免將訓練流程綁定在單一硬體環境。

理論上可以依照相依套件與後端支援情況，部署在不同的 Python 執行環境。

可能的執行環境包括：

- Linux
- Windows
- macOS
- CPU 執行環境
- NVIDIA CUDA 執行環境
- 其他 PyTorch 支援的相容環境

但這些環境並不代表目前都已完成獨立測試。

QLoRA 4-bit 硬體需求

目前 UTE 的 QLoRA 4-bit 實作要求：

- 受支援的 NVIDIA CUDA GPU
- 可正常使用的 PyTorch CUDA 環境
- bitsandbytes
- Accelerate
- 單一 CUDA GPU

QLoRA 透過 4-bit 量化降低 Base Model 的記憶體需求，但不代表不需要 GPU 記憶體。

實際資源需求仍取決於：

- Base Model 大小與架構
- Sequence Length
- Batch Size
- Gradient Accumulation
- LoRA Configuration
- Compute Dtype
- GPU 記憶體容量
- 其他訓練設定

目前的 QLoRA 實作並不代表所有 CUDA GPU、模型架構與套件版本組合都已通過實測。

# 📱 手機與 ARM 環境

UTE 的核心程式使用 Python，不以 x86_64 作為程式本身的必要條件。

因此，ARM 裝置在理論上也可以嘗試執行 UTE。

然而，手機上的 PyTorch、Transformers、PEFT、TRL 相依套件安裝、模型記憶體需求及硬體加速後端，都需要另外驗證。

一般 LoRA 與 QLoRA 的硬體要求不同。

目前 QLoRA 4-bit 實作要求受支援的 CUDA GPU，因此不能直接假設一般 ARM CPU 或手機 GPU 可以使用這項功能。

# ⚠️ 硬體資源提醒

LoRA 可以降低需要訓練的參數量，但基礎模型仍然需要載入記憶體。

QLoRA 透過量化降低 Base Model 的記憶體需求，但仍需要額外空間存放訓練所需的參數、梯度、Optimizer 狀態與其他運算資料。

實際需求取決於：

- Base Model 大小
- Model Precision
- Quantization Configuration
- Sequence Length
- Batch Size
- Gradient Accumulation
- Optimizer
- LoRA Configuration
- CPU / GPU 記憶體容量

請依照實際設備選擇模型與訓練參數。

---

# 🔄 設計原則

UTE 遵循以下設計原則：

1. 模型獨立性
   
   不將引擎綁定至單一模型或 Model ID。

2. 專案獨立性
   
   UTE 是獨立的訓練工具，不屬於任何特定 AI 模型專案。

3. 模組化
   
   硬體偵測、相容性檢查、量化設定、模型載入、LoRA 分析、Embedding 訓練、訓練測試及輸出驗證各自獨立。

4. 可重用性
   
   不同專案可以使用相同的 UTE 訓練流程，而不需要為每個模型重新建立整套訓練引擎。

5. 可驗證性
   
   除了執行訓練，也重視訓練前檢查、整合測試與輸出驗證。

6. 不綁定單一硬體
   
   盡可能將訓練流程與實際硬體環境分離，讓不同環境可以依照相依套件與後端能力使用 UTE。

7. 向後相容性
   
   一般 LoRA 維持預設訓練模式。新增 QLoRA 與 Embedding Training 功能不應強制一般 LoRA 使用者安裝額外的量化依賴或啟用額外訓練模組。

---

# 🗺️ 開發方向

未來可以持續改善的方向包括：

- [ ] 更完整的跨平台相容性驗證
- [ ] 更多模型架構相容性測試
- [ ] 更多 PyTorch 硬體後端測試
- [ ] QLoRA CUDA 端到端測試
- [ ] 更多 GPU 與 bitsandbytes 版本相容性測試
- [ ] 更完整的資料集格式與驗證流程
- [ ] 更完善的訓練中斷與恢復測試
- [ ] 更完整的 Adapter 輸出驗證
- [ ] 依賴版本相容性矩陣
- [ ] 更詳細的硬體資源分析
- [ ] Embedding Training 跨模型架構相容性測試
- [ ] 不同 LoRA Rank 的系統性比較
- [ ] Tiny Model 訓練實驗
- [ ] High-Rank LoRA + Trainable Embeddings 研究
- [ ] Full Fine-Tuning 實驗基準

以上為可能的開發方向，不代表相關功能已經實作。

---

# 🤝 參與專案

歡迎對 Python、PyTorch、Transformers、PEFT、TRL 與模型訓練工具有興趣的開發者參與。

你可以透過以下方式協助：

- 回報 Bug
- 提交 Issue
- 提供相容性測試結果
- 改善文件
- 提交 Pull Request
- 擴充測試模組

在提交修改前，請盡可能確認不會破壞既有的訓練流程與模組介面。

---

# 📜 License

UTE 採用 Apache License 2.0 開源授權。

詳細授權內容請參閱：

"LICENSE"

---

## 🚀 UTE — Universal Training Engine

One Engine. Reusable Training Workflows.

---

## 📬 聯繫創作者

- Gmail: b0953166696@gmail.com
- 本人17歲 🤔 做的不好請見諒
- 獨立開發 ＆ AI 協作
- 緩慢更新 ＆ 除錯
- 純手機 Termux 開發 👀
- 持續開發中…

---

## 👀 作品 & 產品集

- [MySSH](https://github.com/a370373/MySSH-Server)
- [UTE - Universal Training Engine](https://github.com/a370373/UTE-Universal-Training-Engine)
- [0-or-1](https://github.com/a370373/0-or-1)
- [MyDNS](https://github.com/a370373/MyDNS/tree/main)
- [Cyber-Fly-Android-Bridge](https://github.com/a370373/Cyber-Fly-Android-Bridge)
- [My-ADB-Shell](https://github.com/a370373/My-ADB-Shell/tree/main)
- [Cyber-Fly](https://github.com/a370373/Cyber-Fly)
- [MyOS](https://github.com/a370373/MyOS)
- [RWM-1:1 Real World Minecraft](https://github.com/a370373/RWM-Real-World-Minecraft)
- [MyAI - Offline Personal AI Agent System](https://github.com/a370373/MyAI-Offline-Personal-AI-Agent-System-/tree/main)
- [WCL - Web Clone Lab](https://github.com/a370373/web-clone-lab/)
- 持續增加中… 👀

---

## 🤖 AI 協作

UTE 由 a370373/XRH 發起、設計與開發。

開發過程中使用 OpenAI ChatGPT 作為 AI 協作夥伴，協助進行技術分析、程式碼檢查、除錯與文件整理。

產品方向、設計理念與最終決策由專案創作者負責。
