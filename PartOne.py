#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    import re
    sentence = ntlk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    #need to filter out puntctuation 
    words = [word for word in words if word.isalpha()]  # filter out punctuation and numbers
    total_words = len(words)
    total_sentences = len(sentence)     
    total_syllables = sume(count_syl(word, d) for word in words)  # sum the syllables for each word in the text
    if total_words == 0 or total_sentences == 0:  # avoid division by zero
        return 0.0
    fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return fk_grade if fk_grade > 0 else 0.0  # return 0.0 if the grade is negative 
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    import re
    word = word.lower()
    if word.lower() in d:
        return len([p for p in d[word][0] if p[-1].isdigit()]) #this is cheking the sylabeles for words in this dictionary the is digit part is checking the sylable count per secion of the word
    else:
        # Estimate syllables by counting vowel clusters
        vowels = "aeiouy"
        count = 0
        word = word.lower()
        in_vowel_cluster = False
        for char in word:
            if char in vowels:
                if not in_vowel_cluster:
                    count += 1
                    in_vowel_cluster = True
            else:
                in_vowel_cluster = False
        return count if count > 0 else 1
    pass


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    import pandas as pd
    from tqdm import tqdm
    import os
    data = []
    for file in tqdm(os.listdir(path)):
        if file.endswith(".txt"):
            try:
                #extract the data from filename 
                filename_sections = file[:-4].split("-")  # these sections are split by the - 
                if len(filename_sections) == 3: #this is the correct length 
                    title, author, year = filename_sections
                    year= int(year.strip()) #turn year into an integer 
                else: 
                    title, author, year = 'unknown', 'unknown', 'unknown' #this is a fallback if the filename is not in the expected format
                data.append({"title": title, "text": text, "author": author, "year": year})
            except Exception as e:
                print(f"Error in {file}: {e}")
                continue
    dataframe =pd.DataFrame(data)
    dataframe= dataframe.sort_values(by=year, ascending=True) #sort by year
    pass


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    #add new column with parsed doc objects
    df["parsed"] = df["text"].apply(lambda x: nlp(text[:nlp.max_length]))  # parse the text using spaCy
    #save to a pickel file 
    pickle_file = store_path / out_name
    with open(pickle_file, "wb") as f:
        import pickle
        pickle.dump(df, f)
    return df  # return the DataFrame with the parsed column added
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    import nltk
    from nltk import word_tokenize
    newtoken = word_tokenize(text)
    #need to exclude punctuation and ignore case
    # Convert tokens to lowercase and filter out non-alphabetic tokens - this will filter out numbers and punctuation and ignore the lowercase
    newtoken = [token.lower() for token in newtoken if token.isalpha()]
    unique_tokens = set(newtoken) #removes the duplicates
    if len(newtoken) == 0:#if there are no tokens, return 0.0 to avoid division by zero
        return 0.0
    return len(unique_tokens) / len(newtoken) if len(newtoken) > 0 else 0.0


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

