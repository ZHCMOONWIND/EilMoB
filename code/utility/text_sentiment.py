from senticnet.senticnet import SenticNet
import numpy as np
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
sia = SentimentIntensityAnalyzer()
def get_text_sentiment(texts, device):
    sia = SentimentIntensityAnalyzer()
    batch_scores = []

    for text in texts:
        scores = sia.polarity_scores(text)
        score_values = [scores['neg'], scores['neu'], scores['pos'], scores['compound']]
        batch_scores.append(score_values)
    res = torch.tensor(batch_scores, dtype=torch.float32).to(device)
    return res

rake = Rake()
def get_tokens_len(texts, device):
    res = []
    for text in texts:
        text = text.split(' ')
        text = [word.strip() for word in text if len(word.strip()) > 0 and not all(char == '#' for char in word.strip()) and not all(char == '|' for char in word.strip()) and not all(char == '-' for char in word.strip())]
        token_len = len(text)
        text_str = " ".join(text)
        rake.extract_keywords_from_text(text_str)
        keywords = rake.get_ranked_phrases()
        res.append(len(keywords)/token_len)
    res = torch.tensor(res).to(device)
    return res


sn = SenticNet()
def get_word_level_sentiment(texts, tokenizer, device):
    res = []
    max_len = 0
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'][0]
        offset_mapping = encoding['offset_mapping'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        text_res = []
        for idx, (token, (start, end)) in enumerate(zip(tokens, offset_mapping)):
            if idx == 0:
                continue
            if idx == len(offset_mapping) - 1:
                text_res.append(0)
                continue
            if token.startswith("##"):
                text_res.append(0)
            else:
                try:
                    word_polarity_value = float(sn.concept(token)['polarity_value'])
                except KeyError:
                    word_polarity_value = 0.0
                text_res.append(word_polarity_value)
        
        res.append(text_res)
        max_len = max(max_len, len(text_res))
    for i in range(len(res)):
        res[i] += [0] * (max_len - len(res[i]))
    res = torch.tensor(res).to(device)
    return res