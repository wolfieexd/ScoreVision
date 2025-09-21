# 📝 OMR Evaluvator — AI-Powered OMR Scoring System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#license)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-brightgreen)]()

An end‑to‑end Optical Mark Recognition (OMR) system built with Flask + OpenCV. It processes scanned OMR sheets, extracts marked answers, validates against answer keys, supports batch runs, and exports results. Includes a Universal OMR Processor for multi‑format, multi‑orientation inputs.

---

## 🖼️ Home Page (Preview)

<p align="center">
  <img src="https://github.com/wolfieexd/ScoreVision/blob/main/Screenshot.png?raw=true" alt="Picture1" />
</p>

---

## 🔥 Features

- 🚀 Fast, robust bubble detection with OpenCV
- 🧠 Two processors:
  - Standard: tuned for the original sheet format
  - Universal: adaptive to different layouts, sizes, and rotations
- 🧾 Answer key management + validation reporting
- 📦 Batch processing with background status tracking
- 📤 Exports to CSV / Excel / PDF
- 🔎 Quality checks and diagnostics
- 🧪 CLI utilities for validation, comparison, and stress tests

---

## 🗂️ Folder Structure

```
OMR Evaluvator/
├── app.py                        # Flask app entry point
├── app/
│  ├── core/
│  │  ├── omr_processor.py        # Standard OMR Processor
│  │  ├── universal_processor.py   # Universal OMR Processor
│  │  ├── batch_processor.py       # Batch processing engine
│  │  ├── excel_converter.py       # Answer key import/export
│  │  ├── quality_validator.py     # Image/answer quality checks
│  │  └── result_exporter.py       # Exports (CSV/Excel/PDF)
│  ├── static/                     # Frontend assets (css/js/img)
│  └── templates/                  # Flask HTML templates
├── answer_keys/                   # Uploaded/managed answer keys
├── results/                       # Processing outputs
├── production_results/            # Saved demo/production outputs
├── requirements_final.txt         # Fully verified dependencies
├── final_validation.py            # Validation harness (CLI)
├── processor_comparison.py        # Compare processors (CLI)
└── universal_test.py              # Rotation/format stress tests (CLI)
```

---

## ⚙️ Installation (Windows PowerShell)

```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_final.txt
```

> Python 3.11+ recommended (project was verified on 3.13.2).

---

## 🖥️ Run the Web App

```powershell
# From project root
set FLASK_APP=app.py
python app.py
# or
python -m flask run --host=127.0.0.1 --port=5000
```

Open http://127.0.0.1:5000 in your browser.

---

## 🔌 REST API (Selected)

- GET `/` — Home
- GET `/upload`, `/batch`, `/answer-key`, `/validation`, `/results`
- POST `/api/upload-omr` — upload an OMR image
- POST `/api/upload-answer-key` — upload answer key JSON
- POST `/api/process-omr` — process a sheet
  - Body:
    ```json
    {
      "omr_file": "<filename>",
      "answer_key_file": "<filename>",
      "processor_type": "standard|universal"
    }
    ```
- POST `/api/batch-upload`, `/api/batch-process`, `/api/start-batch-processing`
- GET `/api/batch-status/<session_id>`
- GET `/api/list-results`, `/api/validation-reports`
- POST `/api/export-results`
- GET `/api/download/<filename>`
- POST `/api/create-answer-key`

---

## 🔄 Processors

- Standard (default): optimized for the original Img2 sheet format.
- Universal: robust across scans, sizes, light skew/rotation, and differing column placements.

Select Universal via API body:
```json
{
  "omr_file": "your_omr_image.jpeg",
  "answer_key_file": "your_answer_key.json",
  "processor_type": "universal"
}
```

---

## 🧪 CLI Utilities

Run from project root:
```powershell
python final_validation.py          # Validate detection against a small known key
python processor_comparison.py      # Compare Standard vs Universal processors
python universal_test.py            # Stress test rotations and formats
```

---

## 🗝️ Answer Keys

Expected JSON format:
```json
{
  "name": "Sample Key",
  "total_questions": 100,
  "questions": [
    { "question": 1, "answer": "A" },
    { "question": 2, "answer": "D" }
  ]
}
```
Place files in `answer_keys/` for the web UI/API.

---

## 📤 Results & Exports

- Detected answers + diagnostics → `results/`
- Example/production outputs → `production_results/`
- Export CSV/Excel/PDF via the API or UI

---

## 🩺 Troubleshooting

- Ensure the venv is active and dependencies are installed
- Large/corrupt images: try re‑scanning; the Universal processor auto‑resizes
- Orientation/skew: Universal typically recovers without manual rotation
- Import errors: confirm the `app/` folder structure is intact

---

## 🧭 Demo Mode (Roadmap)

Planned enhancements for a full demo experience:
- Demo OMR sheets with known answer patterns
- Guided UI flow and sample datasets
- Synthetic sheet generator
- Interactive tutorials embedded in the web UI

---

## 🧰 Tech Stack

Python • OpenCV • NumPy • SciPy • scikit‑image • Flask • pandas • openpyxl • reportlab

---

## 👤 Maintainers

- Project Maintainer: You/Your Team
- Contributions: PRs and issues welcome for improvements and extensions

---

## 📄 License

This project is intended for evaluation and academic/demo purposes. For commercial use, please contact the maintainers.
