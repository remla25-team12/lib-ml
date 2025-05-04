import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')


def _clean(text):
    """
    Private helper method to clean the text data. 
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    clean_text = ' '.join(text)
    return clean_text


def preprocess_dataset(dataset, max_features=1420, n_rows=900):
    """
    Preprocessing method for the training dataset.

    Args:
        dataset (pd.DataFrame): Training dataset.
        max_features (int): Maximum number of features to use within the count vectorizer.
        n_rows (int): Number of reviews to preprocess from the dataset for training.
    """
    corpus=[]
    for i in range(min(n_rows, len(dataset))):
        clean_review = _clean(dataset['Review'][i])
        corpus.append(clean_review)

    cv = CountVectorizer(max_features = max_features)

    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    return X, y, cv


def preprocess_input(text, vectorizer):
    """
    Preprocessing method for the new input text.
    The same preprocessing steps as in the training dataset are applied using the same count vectorizer model.

    Args:
        text (str): Input text.
        vectorizer (CountVectorizer): Fitted count vectorizer model from training.
    """
    clean_text = _clean(text)
    return vectorizer.transform([clean_text]).toarray()