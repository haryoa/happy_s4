import re
from collections import Counter
from typing import Any, List, Optional, Dict

from torchtext.vocab import vocab


class WordTokenizer:
    """
    Word Tokenizer to tokenize the data into tokens
    """

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        min_freq: int = 1,
        split_pattern: str = r"\s+",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        eos_token: str = "[EOS]",
        sos_token: str = "[SOS]",
        lowercase: bool = False,
    ):
        """
        Word Tokenizer

        Parameters
        ----------
        lowercase: bool
            Lowercase?
        special_tokens: List[str]
            List of special tokens
        min_freq: int
            Minimum frequency of the token, by default = 1
        split_pattern: str
            Tokenizer pattern to distinguish the tokens, by default `\s+` (space)
        """
        self.pad_token = pad_token.lower() if lowercase else pad_token
        self.unk_token = unk_token.lower() if lowercase else unk_token
        self.sos_token = sos_token.lower() if lowercase else sos_token
        self.eos_token = eos_token.lower() if lowercase else eos_token

        sp_token_default = (
            [pad_token, unk_token, eos_token, sos_token]
            if not lowercase
            else [
                pad_token.lower(),
                unk_token.lower(),
                eos_token.lower(),
                sos_token.lower(),
            ]
        )
        self.special_tokens = (
            sp_token_default
            if special_tokens is None
            else sp_token_default + special_tokens
        )
        self.min_freq = min_freq
        self.split_pattern = split_pattern
        self.lowercase = lowercase

        # Fit variables
        self.vocab = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.eos_token_id = None
        self.sos_token_id = None

    def fit(self, *data: List[str]) -> None:
        """
        Fit data's vocabulary

        Parameters
        ----------
        data: List[str]
            The dataset
        """
        all_tokens = []
        for dt in data:
            for x in dt:
                x = x.lower() if self.lowercase else x
                all_tokens += re.split(self.split_pattern, x)
        self.vocab = vocab(
            Counter(all_tokens),
            min_freq=self.min_freq,
            specials=self.special_tokens,
        )
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.sos_token_id = self.vocab[self.sos_token]

    def tokenize_to_ids(self, inp: str, with_eos_sos: bool = False) -> List[str]:
        """
        Tokenize a text input to its input indices

        Parameters
        ----------
        inp: str
            Text input
        with_eos_sos: bool
            Concate it with end of sentence and start of sentence token?

        Returns
        -------
        List[str]
            tokenized text (indices)
        """
        inp_ready = inp.lower() if self.lowercase else inp
        splitted_txt = re.split(self.split_pattern, inp_ready)
        ids_splitted_txt = [
            self.vocab[x] if x in self.vocab else self.vocab[self.unk_token]
            for x in splitted_txt
        ]
        if with_eos_sos:
            ids_splitted_txt = (
                [self.sos_token_id] + ids_splitted_txt + [self.eos_token_id]
            )
        return ids_splitted_txt


def shape_dataset(
    inp_dict: Dict[str, Any],
    word_tokenizer: WordTokenizer,
    text_cols: Optional[List[str]] = None,
    label_col: str = "label",
) -> Dict[str, Any]:
    """
    Used for mapping dataset from text to ids

    Parameters
    ----------
    inp_dict : _type_
        Input dictionary
    word_tokenizer : _type_
        Word tokenizer obj
    text_cols : list, optional
        Text column in the dictionary., by default ["sentence1", "sentence2"]
    label_col : str, optional
        Label column in the dataset, by default "label"

    Returns
    -------
    _type_
        _description_
    """
    if text_cols is None:
        text_cols = ["sentence1", "sentence2"]
    input_ids = word_tokenizer.tokenize_to_ids("[CLS]")
    for col in text_cols:
        input_ids += word_tokenizer.tokenize_to_ids(inp_dict[col])
        input_ids += word_tokenizer.tokenize_to_ids("[SEP]")
        
    # Attention masks for length / masking (if using self-attention)
    attention_masks = [1] * len(input_ids)
    returned_dict = dict(label=inp_dict[label_col], input_ids=input_ids, attention_masks=attention_masks)
    return returned_dict
