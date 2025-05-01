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

def preprocess_dataset(dataset, max_features=1420, n_rows=900):
    corpus=[]
    for i in range(n_rows):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features = max_features)

    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    return X, y


