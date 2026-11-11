# Student Performance Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Create Dataset (1000 students)
np.random.seed(42)
data = {
    'study_hours': np.random.randint(1, 10, 1000),
    'attendance': np.random.randint(50, 100, 1000),
    'previous_score': np.random.randint(30, 100, 1000),
}
df = pd.DataFrame(data)
df['final_score'] = df['study_hours']*4 + df['attendance']*0.3 + df['previous_score']*0.4 + np.random.randint(-5, 5, 1000)
df['final_score'] = df['final_score'].clip(0, 100)

print(df.head())

# 2. Visualization
plt.scatter(df['study_hours'], df['final_score'])
plt.xlabel('Study Hours')
plt.ylabel('Final Score')
plt.title('Study Hours vs Final Score')
plt.show()

# 3. Train Model
X = df[['study_hours', 'attendance', 'previous_score']]
y = df['final_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 4. Accuracy
pred = model.predict(X_test)
print("Accuracy (R2 Score):", round(r2_score(y_test, pred)*100, 2), "%")
print("Prediction for 6 hrs, 85 attendance, 70 prev:", model.predict([[6, 85, 70]]))
