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
if __name__ == "__main__":
    from pathlib import Path
    path = path.cwd() / 'p2-texts' / 'hansard40000.csv'
    df = load_dataset_clean(path)
    print(df.head())  # Display the first few rows of the cleaned dataframe
    print(df['party'].value_counts())  # Display the counts of each party in the cleaned dataframe