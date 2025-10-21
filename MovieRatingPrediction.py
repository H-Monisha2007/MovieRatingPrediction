import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

#Load Dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

#Select relevant columns and drop missing data
df = df[['Name', 'Year', 'Duration', 'Genre', 'Rating', 'Votes', 'Director',
         'Actor 1', 'Actor 2', 'Actor 3']].dropna()

#Clean 'Year' column
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0].astype(int)

#Clean 'Votes' column
df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=True).astype(int)

#Clean 'Duration' column
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)')[0].astype(float)

#Combine actors into a single list column
df['Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].values.tolist()

#Convert 'Genre' to list
df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])

#Encode 'Director' as numeric ID
df['Director'] = df['Director'].astype('category').cat.codes

#Multi-hot encode 'Genre' and 'Actors'
mlb_genre = MultiLabelBinarizer()
mlb_actor = MultiLabelBinarizer()

genre_encoded = pd.DataFrame(mlb_genre.fit_transform(df['Genre']), columns=mlb_genre.classes_)
actor_encoded = pd.DataFrame(mlb_actor.fit_transform(df['Actors']), columns=mlb_actor.classes_)

#Reset indexes to align all DataFrames
df = df.reset_index(drop=True)
genre_encoded = genre_encoded.reset_index(drop=True)
actor_encoded = actor_encoded.reset_index(drop=True)

#Prepare features and target
features = pd.concat([df[['Duration', 'Votes', 'Year', 'Director']], genre_encoded, actor_encoded], axis=1)
target = df['Rating'].astype(float)

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

#Feature importance
importances = model.feature_importances_
feature_names = features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

#Plot top 15 features
top_features = importance_df.head(15)
plt.figure(figsize=(12, 6))
plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()
