# 🌹 RoseMed-27B-BG

```
    ╔══════════════════════════════════════╗
    ║     🌹  RoseMed-27B-BG  🌹          ║
    ║  Bulgarian Medical AI Assistant      ║
    ╚══════════════════════════════════════╝
```

**RoseMed-27B-BG** is a specialized Bulgarian medical AI assistant, fine-tuned on
Bulgarian medical data to serve the Bulgarian healthcare system. Named after
Bulgaria's iconic Rose Valley, RoseMed provides accurate medical information in
Bulgarian — aligned with НЗОК guidelines, the Bulgarian drug registry, and local
clinical protocols.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-org/rosemed/blob/main/RoseMed_Colab.ipynb)

---

## Features

- **Bulgarian medical Q&A** across 10 specialty categories
- **НЗОК-aware** — reimbursement lists, referral system, sick leave
- **Drug information** — brand and generic names used in Bulgaria
- **Symptom checker** with urgency classification
- **REST API** — FastAPI server with rate limiting and CORS
- **Dual inference** — GPU (transformers) or CPU (GGUF) auto-detection

---

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| **Training** | 80 GB VRAM GPU | A100 80GB / H100 |
| **Inference (GPU)** | 24 GB VRAM | RTX 3090 / RTX 4090 |
| **Inference (CPU)** | 16 GB RAM | 32 GB RAM + GGUF Q4 |
| **Disk** | 120 GB free | 200 GB SSD |

---

## Platform Setup

RoseMed scripts are platform-agnostic and run on any GPU cloud provider.

### RunPod

```bash
# Select a pod with A100 80GB
git clone <your-repo> && cd rosemed
bash setup.sh
```

### Vast.ai

```bash
# Search: "A100 80GB" — Ubuntu 22.04 template
git clone <your-repo> && cd rosemed
bash setup.sh
```

### Lambda Labs

```bash
# Launch A100 instance
git clone <your-repo> && cd rosemed
bash setup.sh
```

### Google Colab

```python
# In a Colab notebook with A100 runtime:
!git clone <your-repo>
%cd rosemed
!bash setup.sh
!python 1_download_model.py
# ... continue pipeline steps
```

---

## Quick Start

### 1. Setup

```bash
git clone <your-repo>
cd rosemed
bash setup.sh
```

Edit `.env` with your credentials:

```env
HF_TOKEN=your_huggingface_token_here
BASE_MODEL_ID=your-base-model-id
CHAT_TEMPLATE=your-chat-template-id
WANDB_API_KEY=optional_wandb_key
```

### 2. Pipeline (5 Steps)

```bash
# Step 1: Download base model (~60 GB)
python 1_download_model.py

# Step 2: Generate Bulgarian medical dataset (2000 samples)
python 2_prepare_dataset.py

# Step 3: QLoRA fine-tuning (~8-14 hours on 80GB GPU)
python 3_finetune.py

# Step 4: Merge adapter + export GGUF
python 4_merge_and_export.py

# Step 5: Full WHO + Bulgarian knowledge ingest (RAG + v2 training data)
python 5b_download_bulgarian_pdfs.py   # optional: crawl MoH/NHIF PDFs
python 5c_crawl_medical_web.py 800     # BFS crawl 20+ medical domains + PubMed
python 5_ingest_knowledge_base.py      # merge everything into knowledge base

# Continuous ingest loop (crawl → PDFs → ingest every run)
python 5d_run_full_ingest_loop.py
```

Step 5 builds `data/knowledge_base/chunks.jsonl` for RAG and creates
`data/rosemed_train_full.jsonl` (original 2000 samples + all source Q&A).
For a v2 fine-tune after step 3 finishes:

```bash
cp data/rosemed_train_full.jsonl data/rosemed_train.jsonl
python 3_finetune.py
```

### 3. Run API Server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "rosemed-27b-bg", "version": "1.0.0"}
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Какви са симптомите на диабет?",
    "history": [],
    "language": "bg"
  }'
```

### Symptom Diagnosis

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["главоболие", "температура"],
    "age": 35,
    "gender": "F"
  }'
```

### Medication Info

```bash
curl -X POST http://localhost:8000/medication \
  -H "Content-Type: application/json" \
  -d '{
    "medication_name": "Алфа Норм",
    "query_type": "dosing"
  }'
```

---

## Project Structure

```
rosemed/
├── setup.sh                   # Install dependencies
├── config.py                  # Central configuration
├── 1_download_model.py        # Download base model
├── 2_prepare_dataset.py       # Generate dataset
├── 3_finetune.py              # QLoRA fine-tuning
├── 4_merge_and_export.py      # Merge + GGUF export
├── 5_ingest_knowledge_base.py # WHO + Bulgaria + Orphanet + crawl cache
├── 5b_download_bulgarian_pdfs.py  # Crawl Bulgarian health PDFs
├── 5c_crawl_medical_web.py    # BFS medical web crawler (WHO, NHIF, PubMed…)
├── 5d_run_full_ingest_loop.py # One loop iteration: crawl + ingest
├── crawl_config.py            # Crawl domain whitelist + seeds
├── medical_sources.py         # Source catalog (150+ URLs)
├── server/
│   ├── main.py                # FastAPI server
│   ├── routes.py              # API routes
│   └── schemas.py             # Pydantic schemas
├── tests/
│   └── test_api.py            # API tests
├── data/                      # Dataset files
├── outputs/                   # Model outputs
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Method | QLoRA (4-bit) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Batch size | 1 × 8 grad accum |
| Max sequence length | 4096 |
| Dataset size | 2,000 samples |
| Output model | `rosemed-27b-bg` |

Optional WandB monitoring: set `WANDB_API_KEY` in `.env` (run name: `rosemed-27b-bg`).

---

## Dataset Categories

| Category | Samples |
|----------|---------|
| Обща медицина | 300 |
| Кардиология | 250 |
| Неврология | 200 |
| Онкология | 200 |
| НЗОК и здравна система | 150 |
| Спешна медицина | 150 |
| Лекарства и фармакология | 200 |
| Педиатрия | 150 |
| Психично здраве | 150 |
| Профилактика | 250 |

---

## License

This project is provided for research and educational purposes. See LICENSE for details.

---

## Disclaimer

⚕️ **Тази информация е само за справка. Консултирайте се с лекар.**

RoseMed-27B-BG is an informational AI assistant only. It is **not** intended for
clinical decision-making, diagnosis, or treatment without supervision by a
qualified physician. Always consult a healthcare professional for personal
medical advice. In emergencies, call **112**.
