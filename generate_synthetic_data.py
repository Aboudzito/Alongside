import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
num_responses = 949

# Simulate answers reflecting our global ~86.6% mastery distribution
# Choices: 1 = Driver/Correct, 0 = Passenger/Incorrect
data = {
    'student_id': [f'STU_{i:03d}' for i in range(1, num_responses + 1)],
    'q1_healthy_habit': np.random.choice([1, 0], size=num_responses, p=[0.866, 0.134]),
    'q2_impulse_control': np.random.choice([1, 0], size=num_responses, p=[0.866, 0.134]),
    'q3_perspective_taking': np.random.choice([1, 0], size=num_responses, p=[0.866, 0.134]),
    'q4_accountability': np.random.choice([1, 0], size=num_responses, p=[0.866, 0.134]),
    'q5_mental_health': np.random.choice([1, 0], size=num_responses, p=[0.866, 0.134])
}

df = pd.DataFrame(data)
df.to_csv('synthetic_student_responses.csv', index=False)
print("Synthetic dataset successfully generated (949 responses)!")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df_total.select_dtypes(include='object').columns:
    df_total[column] = label_encoder.fit_transform(df_total[column].astype(str))

df_total.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df_total.drop(columns=['understand_score', 'label'])
y = df_total['understand_score']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ml = RandomForestClassifier(n_estimators=100, random_state=42)
ml.fit(x_train, y_train)
y_pred = ml.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%')

import matplotlib.pyplot as plt

importances = pd.Series(ml.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Top 10 Features (Excluding Target Leakage)')
plt.show()
