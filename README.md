# 📩 SMS Spam Classifier

Classifying SMS messages as spam or ham using NLP. Spam filtering is a real, deployed problem — and Naive Bayes with TF-IDF is still one of the strongest baselines for it because text classification with bag-of-words features tends to have roughly independent word probabilities. Built this as my first NLP project to understand text vectorization before moving to more complex approaches.

## ✨ What I Did and Why
* **TF-IDF Vectorization:** Converted raw text into numerical features using Term Frequency-Inverse Document Frequency with stop word removal. TF-IDF down-weights common words like "the" and "is" that appear everywhere and carry no signal.
* **Multinomial Naive Bayes:** Chose MNB specifically for text — it's designed for count/frequency features and works well with the independence assumption that TF-IDF produces. It also trains fast on small datasets.
* **Asymmetric Error Analysis:** Tracked false positives (ham flagged as spam) and false negatives (spam that slips through) separately. A spam filter that blocks real messages is worse than one that lets some spam through — so minimizing false positives was the priority.
* **Batch Prediction on Custom Input:** Built a pipeline that reads from a `.txt` file and exports results to CSV, making it usable on new data without modifying the notebook.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (TfidfVectorizer, MultinomialNB, train_test_split, confusion_matrix, ConfusionMatrixDisplay)
* **Dataset:** SMS Spam Collection (`spam.csv`) — 5,572 messages, labeled ham or spam (87/13 class split)

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place `spam.csv` and `Spamtest.txt` in the same folder as `main.ipynb`.
4. Run `main.ipynb` cell by cell.

## 📊 Results
* **Train Accuracy:** 98.38% · **Test Accuracy:** 96.68%
* **Confusion Matrix:** 965 True Ham · 113 True Spam · 37 False Ham · 0 False Spam
* **Key Observation:** Zero false positives — the model never incorrectly flags a legitimate message as spam. It misses 37 spam messages out of 150 total, which is an acceptable trade-off. For a spam filter, erring toward not blocking real messages is the right failure mode.

## ⚠️ Limitations
* No preprocessing beyond stop word removal — adding stemming/lemmatization could improve spam recall.
* TF-IDF loses word order entirely. "Not good" and "good not" are treated identically.
* Model wasn't tested on non-English messages or modern spam patterns (URLs, emoji spam).
