#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
nltk.download('punkt') 
nltk.download('punkt_tab') 
from collections import Counter
from math import log
# ...your functions here...

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
    import nltk
    from nltk import word_tokenize
    nltk.download('punkt')

    sentence = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    #need to filter out puntctuation 
    words = [word for word in words if word.isalpha()]  # filter out punctuation and numbers
    total_words = len(words)
    total_sentences = len(sentence)     
    total_syllables = sum(count_syl(word, d) for word in words)  # sum the syllables for each word in the text
    if total_words == 0 or total_sentences == 0:  # avoid division by zero
        return 0.0
    fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    return fk_grade if fk_grade > 0 else 0.0  # return 0.0 if the grade is negative 



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


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    for file in tqdm(list(path.glob("*.txt"))):
        if file.name.endswith(".txt"):
            try:
                #extract the data from filename 
                filename_sections = file.stem.split("-")  # these sections are split by the - 
                if len(filename_sections) == 3: #this is the correct length 
                    title, author, year = filename_sections
                    year= int(year.strip()) #turn year into an integer 
                else: 
                    title, author, year = 'unknown', 'unknown', 'unknown' #this is a fallback if the filename is not in the expected format
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read() #suggestion from co-pilot to fix code bug 
                data.append({"title": title, "text": text, "author": author, "year": year})
            except Exception as e:
                print(f"Error in {file}: {e}")
                continue
    dataframe =pd.DataFrame(data)
    dataframe['year'] = pd.to_numeric(dataframe['year'], errors='coerce')  # convert year to numeric, coerce errors to NaN - suggestion from co-pilot to fix code bug
    dataframe= dataframe.sort_values(by= "year", ascending=True) #sort by year
    return dataframe


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    #add new column with parsed doc objects
    df["parsed"] = df["text"].apply(lambda x: nlp(x[:nlp.max_length]))  # parse the text using spaCy
    store_path.mkdir(parents=True, exist_ok=True)  #create a path 
    #save to a pickel file 
    pickle_file = store_path / out_name
    with open(pickle_file, "wb") as f:
        import pickle
        pickle.dump(df, f)
    return df  # return the DataFrame with the parsed column added



def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    import re
    import nltk
    from nltk import word_tokenize
    nltk.download('punkt')
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
    #step 2 is to calculate the verb poitwise mutual information (PMI) to find strongly associated subjects 
    subjects_counts = Counter() #tracks how often each subject appears
    verb_counts = Counter()  # tracks how often each verb appears
    co_occcurrences = Counter()  # to count how often (subject,verb) pairs appear together
    for token in doc:#loop trhough the doc
        if token.dep_ == "nsubj": #this means its a subject of  verb 
            subject = token.text.lower() #coversts to lowercase for consitency 
            verb = token.head.lemma_.lower()  # this is the verb the subject depends on 
            subjects_counts[subject] += 1  # update the counters 
            verb_counts[verb] += 1
            co_occcurrences[(subject, verb)] += 1  
            #compute teh totals 
    total_subjects = sum(subjects_counts.values())
    total_verbs = sum(verb_counts.values())
    #calculate the PMI scores
    pmi_scores = []  # list to store the PMI scores
    for (subject, verb), count in co_occcurrences.items():
        prob_subject = subjects_counts[subject] / total_subjects   #probability od subject appearing
        prob_verb = verb_counts[verb] / total_verbs  # probability of verb appearing
        prob_coocurance = count / len(doc)  # probability of subject and verb appearing together

        pmi_scores=log(prob_coocurance / (prob_subject * prob_verb)) if prob_subject > 0 and prob_verb > 0 else 0
        pmi_scores.append((subject, verb, pmi_scores))  # append the subject, verb and pmi score to the list
    return sorted(pmi_scores, key=lambda x: x[2], reverse=True)[:10]  #gives the top 10 subjects with highest PMI




def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
        #step 1 is to find the verb in the document
    subject_counter =[]#list to collect the subjects 
    for token in doc:
        if token.dep_ == "nsubj" and token.head.lemma_ == verb:
            #this is the subject of the verb
            subject_counter.append(token.text.lower()) #add the subject to the list
    return subject_counter.most_common(10)#this will return the 10 most common subjects
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjective = [token.text.lower() for token in doc if token.pos_ == "ADJ"]  # get all adjectives
    adjective_counter = Counter(adjective)  # count the occurrences of each adjective
    return adjective_counter.most_common(10)  # return the 10 most common adjectives


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    path = Path.cwd() / "Part1_novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    print(adjective_counts(df))
    
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")


