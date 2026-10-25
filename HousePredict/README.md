# HousePredict – Property Valuation Demo

A dual‑front‑end demo that predicts house prices (King County, WA) using a simple
Linear Regression model trained on the classic “kc_house_data.csv”.

## Features

- **Streamlit UI** (primary) – modern look with larger input labels, gradient
  prediction card, and a celebratory balloon animation.
- **Node/Express UI** (secondary) – static HTML form that talks to the same
  Python inference script via `python-shell`.
- Python pipeline (`scikit‑learn`) fully reproducible with `train_model.py`.
- Docker‑ready (you can add a `Dockerfile` later) and ready for CI/CD.

## Quick start (Streamlit)

```bash
# Install Python deps
pip install -r HousePredict/requirements.txt

# Train the model (optional – a pre‑trained .joblib is already in the repo)
python HousePredict/model/train_model.py

# Run the UI
streamlit run HousePredict/app.py
```

Open http://localhost:8501 in your browser.

## Quick start (Node/Express)

```bash
cd HousePredict
npm install
npm start
```

Open http://localhost:3000 in your browser.

## Project structure

```
HousePredict/
├─ app.py               # Streamlit UI
├─ model/
│   ├─ train_model.py   # training script
│   └─ predict.py       # inference wrapper
├─ public/              # static HTML UI assets
│   ├─ index.html
│   ├─ style.css
│   └─ script.js
├─ requirements.txt
├─ package.json
├─ .gitignore
└─ data/                # (optional) raw CSV files
```

## License

MIT – feel free to fork, adapt, and deploy.
