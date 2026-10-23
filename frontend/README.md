# Academic Project Novelty Detection — React + Vite Frontend

A modern, responsive Explainable AI (XAI) dashboard for evaluating academic research proposals against a 600-project reference corpus.

## Tech Stack
- **Framework:** React 19 + Vite 6+
- **Styling:** Tailwind CSS v4
- **Charts:** Recharts
- **Icons:** Lucide React
- **Backend API:** FastAPI (`http://127.0.0.1:8000`)

---

## Getting Started

### 1. Start the FastAPI Backend
From the repository root:
```bash
python -m uvicorn src.api.main:app --reload --port 8000
```
The API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Start the React Frontend
In a new terminal, navigate to the `frontend` folder:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Features
- **Dashboard Overview:** Reference corpus status (600 baseline historical projects), submitted proposal count, model operational status, and recent submissions.
- **8-Section Proposal Evaluation:** Guided form for all eight required academic sections with field-level validation, sample demonstration pre-fill presets, and loading states.
- **Explainable AI (XAI) Novelty Report:**
  - Radial novelty percentage gauge and assessment tier badge.
  - Prominent mandatory academic disclaimer.
  - Recharts section cosine similarity bar chart.
  - Recharts horizontal Ridge feature contributions bar chart.
  - Top 5 closest historical projects with section overlap/differentiation breakdown and verified snippet keyword alignment.
- **Deferred Save Flow:** Scoring calls `/api/score` without writing to the database. Saving is an explicit action calling `/api/save`. Editing any field after evaluation invalidates the report and requires re-evaluation.
- **Proposal History:** Paginated table (`limit`/`offset`) with live search and detailed full-report modal view.
