# 📩 SMS Spam Classifier
Classifying SMS messages as spam or ham using TF-IDF vectorization and a Multinomial Naive Bayes model, with confusion matrix evaluation and support for batch predictions on custom text files.

## ✨ Features
* **Exploratory Data Analysis:** Checks for null values and inspects the dataset structure, with label encoding to convert `ham`/`spam` labels into numeric values for model compatibility.
* **TF-IDF Vectorization:** Converts raw SMS text into numerical features using Term Frequency-Inverse Document Frequency, with English stop word removal and lowercasing for cleaner input.
* **Naive Bayes Classification:** Trains a Multinomial Naive Bayes model — a fast and effective algorithm for text classification tasks.
* **Model Evaluation:** Measures accuracy on both training and test sets, and visualizes performance using a labeled Confusion Matrix display.
* **Batch Prediction on Custom Input:** Reads messages line-by-line from a `.txt` file, runs predictions on each, and exports results to a CSV for easy review.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (TfidfVectorizer, MultinomialNB, train_test_split, confusion_matrix, ConfusionMatrixDisplay)
* **Dataset:** SMS Spam Collection (`spam.csv`) — labeled messages with two columns: `Class` (ham/spam) and `Message`

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place `spam.csv` in the same folder as `main.ipynb`.
4. Run `main.ipynb` cell by cell.
5. Optionally place a `Spamtest.txt` file (one message per line) in the same folder to get batch predictions in `Prediction_test.csv`!

## 📊 Results
* **Model:** Multinomial Naive Bayes with TF-IDF features
* **Train Accuracy:** ~98.6% · **Test Accuracy:** ~97.2%
* **Key Observation:** The model performs very well on standard spam patterns but may struggle with newer spam techniques that mimic natural language. Performance is also dependent on the quality and diversity of the training data.
