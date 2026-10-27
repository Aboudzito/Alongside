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