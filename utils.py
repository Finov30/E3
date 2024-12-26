import re
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sparse import COO


def preprocess_text(text):
    """Nettoie et formate le texte en conservant les noms de produits."""
    text = text.lower()
    # Conserver les noms de produits entre accolades
    product_names = re.findall(r'\{(.*?)\}', text)
    text = re.sub(r'\{(.*?)\}', '', text) # Supprimer les accolades
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Supprimer ponctuation etc.
    words = text.split()
    # Supprimer les stopwords et appliquer le stemming
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    # Réintégrer les noms de produits
    words.extend(product_names)
    return ' '.join(words)


def prepare_data(df):
    # Appliquer le prétraitement optimisé
    df['Ticket Description'] = df['Ticket Description'].apply(preprocess_text)
    X = df['Ticket Description']
    y = df['Ticket Type']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded) # Stratification ajoutée

    return X_train, X_test, y_train, y_test, label_encoder



def lazy_evaluate_models(X_train, X_test, y_train, y_test):
    """Entraîne et évalue plusieurs modèles avec LazyPredict."""

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Convertir les matrices creuses en DataFrames creux avec COO
    X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_vec, columns=vectorizer.get_feature_names_out())
    X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test_vec, columns=vectorizer.get_feature_names_out())


    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train_df, X_test_df, y_train, y_test)



    return models, predictions, vectorizer