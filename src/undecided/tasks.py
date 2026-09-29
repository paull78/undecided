"""Per-dataset wording: instruction templates and label paraphrases.

The first instruction of each list is the canonical one, used by loaders and evaluation.
Augmentation picks among all of them at training time. NLI templates take the hypothesis.
"""

INSTRUCTIONS = {
    "ag_news": [
        "What is this news article about?",
        "Classify the topic of this news article.",
        "Which section of a newspaper does this story belong to?",
        "Pick the category that best fits this headline and summary.",
        "What subject does this article cover?",
        "Assign a topic to this news story.",
        "Which news category is this?",
        "Categorise this article by topic.",
    ],
    "emotion": [
        "Which emotion does the writer express?",
        "How is the author feeling?",
        "Identify the emotion in this text.",
        "What feeling comes through in this message?",
        "Classify the emotion of this sentence.",
        "Which emotion best describes this post?",
        "What is the writer's emotional state?",
        "Label the dominant emotion.",
    ],
    "banking77": [
        "What does this banking customer want?",
        "Classify the intent of this message to a bank.",
        "What is the customer asking the bank about?",
        "Route this banking support request to the right topic.",
        "Which banking issue is this message about?",
        "Identify the customer's request.",
        "What is the purpose of this customer query?",
        "Pick the intent that matches this banking question.",
    ],
    "trec": [
        "What kind of answer does this question expect?",
        "Classify the type of this question.",
        "What is this question asking for?",
        "Which answer type fits this question?",
        "Categorise this question by the answer it needs.",
        "What sort of information would answer this?",
        "Label the expected answer type.",
        "What category does the answer to this question fall into?",
    ],
    "sst2": [
        "What is the sentiment of this review?",
        "How does the reviewer feel about the film?",
        "Classify the sentiment of this sentence.",
        "Is the tone of this review favourable?",
        "What opinion does this movie review express?",
        "Rate the sentiment of this text.",
        "Label the sentiment.",
        "What is the reviewer's overall attitude?",
    ],
    "mnli": [
        'Given the text, is this statement true: "{hypothesis}"',
        'Does the text support the claim "{hypothesis}"?',
        'Based only on the text, is it true that "{hypothesis}"?',
        'Read the text. Statement: "{hypothesis}". Is the statement true, false, or impossible to tell?',
        'Can we conclude "{hypothesis}" from the text?',
        'Is "{hypothesis}" implied by the passage?',
    ],
}

# A few alternatives per dataset. Augmentation swaps an option for one of these at random,
# so the model can't rely on memorising exact label strings.
PARAPHRASES = {
    "ag_news": {
        "world": ["world news", "international affairs"],
        "sports": ["sport", "athletics and games"],
        "business": ["economy", "finance"],
        "science and technology": ["science", "technology", "sci-tech"],
    },
    "emotion": {
        "sadness": ["sad", "sorrow"],
        "joy": ["happiness", "happy"],
        "love": ["affection", "romance"],
        "anger": ["angry", "rage"],
        "fear": ["scared", "anxiety"],
        "surprise": ["surprised", "astonishment"],
    },
    "sst2": {
        "negative": ["bad", "unfavourable"],
        "positive": ["good", "favourable"],
    },
    "mnli": {
        "true": ["yes", "entailed"],
        "cannot tell": ["maybe", "unknown"],
        "false": ["no", "contradicted"],
    },
    "banking77": {
        "card arrival": ["when will my card arrive"],
        "lost or stolen card": ["missing card"],
        "exchange rate": ["currency conversion rate"],
        "top up failed": ["failed top-up"],
        "pin blocked": ["locked pin"],
        "transfer not received by recipient": ["recipient didn't get the transfer"],
    },
    "trec": {
        "an individual": ["a person"],
        "cities": ["a city"],
        "countries": ["a country"],
        "dates": ["a date", "when something happened"],
        "definition of something": ["a definition"],
        "abbreviation": ["an acronym"],
    },
}
