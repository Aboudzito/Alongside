# Alongside

**A peer-voiced AI companion for students, built on a real school character-education program.**

Alongside helps students talk through everyday situations (friendship conflict, feeling judged, motivation, boundaries) in the voice of a supportive friend rather than a lecturer. Its conversation framework comes from *Dare To Be Aware*, a peer-led, 11-session character-building program I helped run at The Egyptian Language School (ELS), and from the survey feedback we collected after each session.


---

## Background: Dare To Be Aware

Student teens taught other student teens through 11 face-to-face sessions covering accountability, mental health literacy, empathy and perspective-taking, gratitude, communication, boundaries, social awareness, healthy habits, and more. After each session we ran a short survey (quiz questions, plus ratings of the session and presenters).

**Real data (private).** The survey data consists of **949 responses across 11 sessions**. It belongs to the students and the school, so it is **not published in this repository**. Across the sessions, about **86.6% of responses showed the intended understanding** of the lesson objectives. This figure is descriptive: it comes from self-reported, post-session surveys.

**What this data can and can't show.** The surveys were taken once, after each session, with no pre-session baseline. They indicate how well lessons were understood *at that moment*. They can't prove that students' mindsets changed over time. Measuring that would need pre/post surveys.

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend: serves the frontend and a `/chat` endpoint backed by Google Gemini |
| `Fair.html` | Frontend chat/demo interface |
| `generate_synthetic_data.py` | Creates a **synthetic** dataset with the same structure as the real surveys |
| `train_model.py` | Trains and evaluates a classifier with a baseline, cross-validated tuning, and a held-out test set |
| `render.yaml`, `requirements.txt` | Deployment config for Render |

## ML pipeline (on synthetic data)

Because the real data is private, the pipeline is demonstrated on **synthetic data**. Students in the synthetic set share a hidden "engagement" trait that influences their quiz answers, ratings, and understanding, so the columns are correlated. **The numbers below show that the pipeline works, not real findings about students.**

```bash
python generate_synthetic_data.py   # creates synthetic_student_responses.csv
python train_model.py               # trains, tunes, evaluates, saves model
```

What `train_model.py` does:

- Predicts whether a student **understood** a session from quiz answers, session/presenter ratings, and topic
- Uses a **stratified 80/20 train/test split** (the target is imbalanced, about 87/13)
- Compares against a **majority-class baseline** and a logistic regression
- Tunes a Random Forest with **5-fold cross-validated randomized search**, optimizing macro-F1
- Reports precision, recall, F1, a confusion matrix, and feature importances

**Why not just report accuracy?** With about 87% of students in the "understood" class, a model that always says "understood" gets about 87% accuracy while learning nothing. On the synthetic data, the tuned Random Forest has *lower* accuracy than that baseline (about 77% vs 87%) but a much better macro-F1 (about 0.64 vs 0.46), because it actually identifies students who didn't understand. Macro-F1 and per-class recall are the fairer measures here.

## Run locally

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env     # never commit this file
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000.

## Tech stack

Python, FastAPI, Uvicorn, scikit-learn, pandas, NumPy, Matplotlib, Google Gemini API, HTML/CSS/JS, Render (deployment).

## Limitations & safety

- Alongside is a conversational companion, **not** a counselor or mental-health service. If a student is in distress, it encourages them to talk to a trusted adult or counselor.
- Survey data is self-reported and was collected from one school, so results may not generalize.
- ML results in this repo come from synthetic data and are for demonstration only.
- API keys live in `.env` and are excluded from version control.

## Author

**Abed Habib**: [GitHub](https://github.com/Aboudzito)
