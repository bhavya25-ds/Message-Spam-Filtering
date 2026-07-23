# 📩 SMS Spam Classifier
Classifying SMS messages as spam or ham using NLP. Spam filtering is a real, deployed problem — and Naive Bayes with TF-IDF is still one of the strongest baselines for it because text classification with bag-of-words features tends to have roughly independent word probabilities. Built this as my first NLP project to understand text vectorization before moving to more complex approaches.

## ✨ What I Did and Why
* **TF-IDF Vectorization:** Converted raw text into numerical features using Term Frequency-Inverse Document Frequency with stop word removal and bigrams (`ngram_range=(1,2)`). TF-IDF down-weights common words like "the" and "is" that carry no signal; bigrams let the model pick up short phrases like "free prize" or "claim now" that single words miss.
* **Complement Naive Bayes:** Switched from MultinomialNB to ComplementNB after noticing spam recall was too low (72%). ComplementNB handles class imbalance better by training on the complement of each class, pushing spam recall up to ~89%.
* **Train/Test Split with Stratification:** Evaluated on a held-out 20% test set with stratified sampling to preserve the 87/13 class ratio. Refitted on full data after evaluation so the deployed model uses everything available.
* **Batch Prediction on Custom Input:** Built a pipeline in the notebook that reads from a `.txt` file and exports results to CSV, making it usable on new data without modifying the notebook.
* **Streamlit Web App:** Deployed an interactive interface where users can paste any message and get an instant spam/ham prediction with confidence score.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (TfidfVectorizer, ComplementNB, train_test_split, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
* **App:** Streamlit
* **Dataset:** SMS Spam Collection (`spam.csv`) — 5,572 messages, labeled ham or spam (87/13 class split)

## 🚀 Setup

**To run the notebook:**
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place `spam.csv` and `Spamtest.txt` in the same folder as `main.ipynb`.
4. Run `main.ipynb` cell by cell.

**To run the Streamlit app locally:**
1. Install dependencies: `pip install streamlit pandas scikit-learn`
2. Place `spam.csv` in the same folder as `app.py`.
3. Run: `streamlit run app.py`

Or try the **[live demo]([#](https://spammessage-classifier.streamlit.app/))** 

## 📊 Results
*(From notebook — MultinomialNB baseline)*
* **Train Accuracy:** 98.38% · **Test Accuracy:** 96.68%
* **Confusion Matrix:** 965 True Ham · 113 True Spam · 37 False Ham · 0 False Spam

*(Deployed app — ComplementNB with bigrams)*
* **Test Accuracy:** 97.4% · **Spam Recall:** 89%
* **Key Observation:** Switching to ComplementNB cut missed spam roughly in half compared to the MultinomialNB baseline, at the cost of a small increase in false positives — the right trade-off for a spam filter.

## ⚠️ Limitations
* No preprocessing beyond stop word removal — adding stemming/lemmatization could improve spam recall further.
* TF-IDF loses word order entirely. "Not good" and "good not" are treated identically.
* Model wasn't tested on non-English messages or modern spam patterns (URLs, emoji spam).
* Training data is from ~2005. Modern spam phrasing may not match learned patterns well.
