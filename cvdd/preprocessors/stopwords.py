from typing import List, Optional

import nltk
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer
from xallennlp.data.preprocessors import Preprocessor


@Preprocessor.register("stopwords")
class Stopwords(Preprocessor[str, str]):  # type: ignore[misc]
    def __init__(
        self,
        stopwords: Optional[List[str]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        super().__init__()
        nltk.download()
        self._stopwords = set(stopwords or nltk.corpus.stopwords.words("english"))
        self._tokenizer = tokenizer or WhitespaceTokenizer()

    def __call__(self, data: str) -> str:
        tokens = [
            token.text
            for token in self._tokenizer.tokenize(data)
            if token.text and token.text not in self._stopwords
        ]
        return " ".join(tokens)
