import pandas as pd
#2(a)(i): load the dataset and clean the data replacing labour (co-op) with labour 
def load_dataset_clean(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
    df['party']= df['party'].replace('Labour (Co-op)', 'Labour')
    #2(a)(ii): remove the rows that are not part of top 4 parties 
    top_four = df['party'].value_counts().nlargest(4).index
    df = df[df['party'].isin(top_four)] and (df['party'] != 'Speaker')
    #2 (a)(iii): remove the rows were the speech_class column is not speech
    df = df[df['speech_class'] == 'speech']
    #2(a)(iv): remove any rows where speech column is less than 1000 characters
    df = df[df['speech'].str.len() >= 1000] 
    #2 (a)(v) return the cleaned dataframe
    return df
#2(b) vectorise the speech column using the TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklern.model_selection import train_test_split
def vectorise_speech(df):#function to vecotise speech column
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)#set max features to 3000
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    #split the data into training and testing sets with random seed 26
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
    return X_train, X_test, y_train, y_test, vectorizer
#2(c) train Random Forest Classifier and SVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
def train_random_forest(X_train, X_test, y_train, y_test):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
    return X_train, X_test, y_train, y_test, vectorizer

if __name__ == "__main__":
    from pathlib import Path
    path = path.cwd() / 'p2-texts' / 'hansard40000.csv'
    df = load_dataset_clean(path)
    print(df.head())  # Display the first few rows of the cleaned dataframe
    print(df['party'].value_counts())  # Display the counts of each party in the cleaned dataframe