# Virtual Machine Learning Laboratory

A local, interactive Streamlit application for exploring machine learning concepts. Each experiment lives on its own page, and students can switch between them using the sidebar navigation.

## Experiments

- **Hidden Markov Models (`pages/1_HMM_Lab.py`)** — infer a hidden weather sequence from activities. View a state-transition diagram, transition and emission probabilities, and step-by-step Viterbi candidate calculations. The generated hidden states are available in a collapsible answer key.
- **k-Nearest Neighbors (`pages/2_KNN_Lab.py`)** — adjust the number of neighbors and data split, then view the decision regions and test accuracy for a synthetic classification dataset.

## Requirements

- Python 3.10 or newer
- Git (optional for running the app; needed to publish it to GitHub)

## Run on your computer

Open PowerShell in this folder and create an isolated Python environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`) to open in a browser. The app and experiment data run on the student's computer.

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` for the first two commands; the remaining commands are the same.

If you already created `.venv`, activate it and run `pip install -r requirements.txt` after project updates so newly added packages are installed.

## Add this project to GitHub

Create an empty repository on GitHub, then run these commands from this folder. Replace the URL with the repository URL GitHub provides.

```bash
git init
git add .
git commit -m "Add virtual machine learning experiments"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Do not commit `.venv` or credentials. For later updates, use `git add .`, `git commit -m "Describe the update"`, and `git push`.
