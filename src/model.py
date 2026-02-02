import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/flipkart_reviews.csv")
df = df[df['Rating'] != 3]
df['sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

negative = df[df['sentiment'] == 0]

cv = CountVectorizer(max_features=15)
X_neg = cv.fit_transform(negative['Review Text'])

words = cv.get_feature_names_out()
counts = X_neg.sum(axis=0).A1

pain_points = dict(zip(words, counts))
print(pain_points)
