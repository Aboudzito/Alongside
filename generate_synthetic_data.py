"""
Generate a SYNTHETIC student-survey dataset for demonstration.

The real "Alongside" survey data belongs to the students and the school
and is NOT published. This script produces fake data with a *realistic structure*
so the training pipeline can be run and reviewed by anyone.

How it works
------------
Each synthetic student gets a hidden "engagement" level. That hidden level
nudges their quiz answers, their ratings of the session, and whether they
understood the lesson. Because the columns share a common cause, they are
correlated, so a model has real patterns to learn (unlike independent coin flips).

The overall understanding rate is calibrated to ~86.6% to mirror the real
aggregate result. Nothing here is real student data.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 949
TARGET_RATE = 0.866

TOPICS = [
    "Accountability", "Communication Skills", "Critical Thinking",
    "Diversity & Inclusion", "Empathy & Sympathy", "Gratitude",
    "Healthy Habits", "Mental Health", "Boundaries", "Expectations",
    "Social Awareness",
]

rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrate_intercept(signal, target_rate):
    """Find the intercept so that mean(sigmoid(intercept + signal)) == target_rate."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(mid + signal).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


engagement = rng.normal(0, 1, N)                      # hidden trait
topic_idx = rng.integers(0, len(TOPICS), N)
topic_difficulty = rng.normal(0, 0.3, len(TOPICS))[topic_idx]

# Quiz answers: 1 = correct, 0 = incorrect
quiz = {
    f"q{i}_correct": rng.binomial(1, sigmoid(1.6 + 0.9 * engagement - topic_difficulty))
    for i in range(1, 6)
}


def rating(base, strength):
    """1-5 rating that rises with engagement, plus noise."""
    raw = base + strength * engagement + rng.normal(0, 0.8, N)
    return np.clip(np.round(raw), 1, 5).astype(int)


session_score = rating(4.0, 0.6)
presenter_score = rating(4.1, 0.4)
presenter_engaging_score = rating(3.9, 0.5)

quiz_mean = np.mean(list(quiz.values()), axis=0)
signal = 1.2 * engagement + 1.5 * (quiz_mean - 0.8) - topic_difficulty
intercept = calibrate_intercept(signal, TARGET_RATE)
understood = rng.binomial(1, sigmoid(intercept + signal))

df = pd.DataFrame({
    "student_id": [f"STU_{i:03d}" for i in range(1, N + 1)],
    "topic": [TOPICS[i] for i in topic_idx],
    **quiz,
    "session_score": session_score,
    "presenter_score": presenter_score,
    "presenter_engaging_score": presenter_engaging_score,
    "understood": understood,
})

df.to_csv("synthetic_student_responses.csv", index=False)
print(f"Saved {len(df)} synthetic rows. Understanding rate: {df['understood'].mean():.1%}")
