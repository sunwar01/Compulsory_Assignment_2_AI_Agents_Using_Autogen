import random
from typing import Annotated, List

SENTIMENT_VALUES = {"positive", "negative", "neutral"}

def sentiment_analysis(text: Annotated[List[str], "A list of strings"]) -> Annotated[List[int], "A list of integers"]:
    return [
        random.randint(0, 2)
        for _ in range(len(text))
    ]
