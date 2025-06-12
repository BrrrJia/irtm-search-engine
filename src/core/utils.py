from nltk.corpus import stopwords
import re
import emoji
import unicodedata
import pandas as pd
import csv

EN_STOP = set(stopwords.words("english"))
DE_STOP = set(stopwords.words("german"))
ALL_STOPWORDS = EN_STOP | DE_STOP  # union of both sets


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[newline\]|\[tab\]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"@\w+", " ", text)  # remove mentions
    text = re.sub(r"#", "", text)  # keep hashtags as words
    text = emoji.replace_emoji(text, replace=" ")  # remove emojis
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = "".join(
        c for c in text if unicodedata.category(c)[0] != "S"
    )  # filter out symbol characters
    return text.strip()


def tokenize(text):
    return re.findall(r"\b[\wäöüß]{2,}\b", text)  # only alphabetic tokens, min 2 chars


def filter_stopwords(tokens):
    return [t for t in tokens if t not in ALL_STOPWORDS]


def preprocess(text):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = filter_stopwords(tokens)
    return tokens


def process_row(df):
    for _, row in df.iterrows():
        text = row["text"]
        doc_id = row["docID"]
        terms = preprocess(text)
        yield (terms, doc_id)


def read_data(filename, cls=False, limit=None):
    if cls:
        df = pd.read_csv(
            filename,
            sep="\t",
            header=None,
            names=["name", "label", "title", "review"],
            on_bad_lines="warn",
            encoding="utf-8",
            engine="python",
            quoting=csv.QUOTE_NONE,
        )
        return df
    elif "game" in filename:
        df = pd.read_csv(
            filename,
            sep="\t",
            header=None,
            names=["name", "label", "title", "review"],
            on_bad_lines="warn",
            encoding="utf-8",
            engine="python",
            quoting=csv.QUOTE_NONE,
        )
        df["text"] = df[["name", "title", "review"]].fillna("").agg(" ".join, axis=1)
        df = df.drop(["name", "title", "review"], axis=1)
    elif "tweets" in filename:
        df = pd.read_csv(
            filename,
            sep="\t",
            header=None,
            names=["tweetID", "text"],
            usecols=[1, 4],
            on_bad_lines="warn",
            encoding="utf-8",
            engine="python",
            quoting=csv.QUOTE_NONE,
        )
        df = df.drop_duplicates(subset="tweetID", keep="first").reset_index(
            drop=True
        )  # drop the duplicated lines

    df["docID"] = list(range(1, len(df) + 1))
    return df.iloc[:limit]
