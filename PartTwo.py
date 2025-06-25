import pandas as pd

#2(a)(i): load the dataset and clean 
def load_dataset_clean(path):
    try:
        df = pd.read_csv(path)
        df['party']= df['party'].str.strip().replace('Labour (Co-op)', 'Labour')
        # remove speaker 
        df = df[df['party'] != 'Speaker']
    #2(a)(ii): remove the rows that are not part of top 4 parties 
        top_four = df['party'].value_counts().nlargest(4).index
        df = df[df['party'].isin(top_four)] 
    #2 (a)(iii): remove the rows were the speech_class column is not speech
        df = df[df['speech_class'] == 'Speech']
    #2(a)(iv): remove any rows where speech column is less than 1000 characters
        df = df[df['speech'].str.len() >= 1000] 
        print(f'Data shape:{df.shape}')  # Print the shape of the dataframe after cleaning
    #2 (a)(v) return the cleaned dataframe
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None 
#2(b) vectorise the speech column using the TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
def vectorise_speech(df):#function to vecotise speech column
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)#set max features to 3000
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    #split the data into training and testing sets with random seed 26
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=26)
    return X_train, X_test, y_train, y_test, vectorizer
#2(c) train Random Forest Classifier and SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
def train_models(X_train, X_test, y_train, y_test):
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=26),
        'SVM': SVC(kernel='linear', random_state=26)}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Results for {name}:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro')}")
#2(d) Adjust the parameters of TfidfVectorizer so that unigram, bigram and trigram are used
def vectorise_speech_with_ngrams(df):   
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=3000)
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    #split the data into training and testing sets with random seed 26
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=26)
    #train the classifeers with the new vectorised data
    classifiers ={ 'Random Forest with ngram': RandomForestClassifier(n_estimators=300, random_state=26)
    , 'SVM with ngram':SVC(kernel='linear', random_state=26)}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Results for {name} with ngrams (1,3):")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro')}")
 #2(e) implement new custom tonkenizer - this can be done anyway you like
import re
import nltk 
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
def custom_tokenizer(text):
    # Remove punctuation and tokenize
    texts = re.sub(r'[^\w\s]', '', text) #remove punctuation 
    tokens = word_tokenize(texts.lower())  # Tokenize and convert to lowercase
    return tokens
    # Remove stop words
def vectorise_speech_using_custom_tokeniser(df):
    vectoriser = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer, max_features=3000)
    X = vectoriser.fit_transform(df['speech'])
    y = df['party']
    #split the data into training and testing sets with random seed 26
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=26)
    #train the classifiers with the new vectorised data
    classifiers = {
        'Random Forest with custom tokenizer': RandomForestClassifier(n_estimators=300, random_state=26),
        'SVM with custom tokenizer': SVC(kernel='linear', random_state=26)}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Results for {name} with custom tokenizer:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro')}")



if __name__ == "__main__":
    from pathlib import Path
    path = Path.cwd() / 'p2-texts' / 'hansard40000.csv'
    df = load_dataset_clean(path)
    print(f' Cleaned data preview: {df.head()}')  # Display the first few rows of the cleaned dataframe
    X_train, X_test, y_train, y_test, vectorizer = vectorise_speech(df)
    train_models(X_train, X_test, y_train, y_test)
    vectorise_speech_with_ngrams(df)
    vectorise_speech_using_custom_tokeniser(df)

