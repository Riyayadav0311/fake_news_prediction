import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
import joblib
import nltk

# Download stopwords
nltk.download('stopwords')

# Load and label data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine title + text
fake["combined"] = fake["title"] + " " + fake["text"]
true["combined"] = true["title"] + " " + true["text"]

# Merge and shuffle
df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# Show class balance
print("\n✅ Class distribution:")
print(df["label"].value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["combined"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    max_df=0.7
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "fake_news_model_nb.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer_nb.joblib")
print("\n✅ Model + vectorizer saved successfully!")
