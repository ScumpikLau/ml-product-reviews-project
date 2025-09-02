import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("ml-product-reviews-project/data/IMLP4_13-product_reviews_full.csv")

# Drop all rows with missing values
df = df.dropna()

# Step 1: Convert all sentiment values to lowercase and strip extra spaces
df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

df['sentiment'] = df['sentiment'].astype('category')

df = df.drop(columns=['review_uuid', 'product_name', 'product_price'])

# Create new column with length of each review text
df['review_length'] = df['review_text'].astype(str).str.len()

# Features and label
x = df[['review_title', 'review_text', 'review_length']]
y = df['sentiment']

preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "review_title"),
        ("text", TfidfVectorizer(), "review_text"),
        ("length", MinMaxScaler(), ["review_length"])
    ]
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier())
])

# Train the model on the entire dataset
pipeline.fit(x, y)

# Save the model to a file
joblib.dump(pipeline, "ml-product-reviews-project/model/sentiment_model.pkl" )

print("Model trained and saved as 'ml-product-reviews-project/model/sentiment_model.pkl' ")