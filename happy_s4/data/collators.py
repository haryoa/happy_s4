"""
Collators goes here
"""
import torch


class BatchCollators:
    """
    Batch collators for Pytorch
    """

    def __init__(self, pad_token_ids):
        self.pad_token_ids = pad_token_ids
        self.pad_strategy = "max_length"

    def _pad_helper(self, x, max_length):
        cur_len = len(x)
        added_pad_len = max_length - cur_len
        padded_x = x + [self.pad_token_ids] * added_pad_len
        return padded_x

    def _pad_self(self, input_ids):
        """
        Pad the seq to max length
        """
        max_length = max(map(len, input_ids))
        padded_input_ids = [self._pad_helper(x, max_length) for x in input_ids]
        return padded_input_ids

    def __call__(self, inp):
        input_ids, labels, attention_masks = [], [], []
        for i in inp:
            input_ids.append(i["input_ids"])
            labels.append(i["label"])
            if 'attention_masks' in i:
                attention_masks.append(i["attention_masks"])
        input_ids = self._pad_self(input_ids)
        attention_masks = self._pad_self(attention_masks)
        return {
            "input_ids": torch.LongTensor(input_ids),
            "label": torch.LongTensor(labels),
            "attention_masks": torch.LongTensor(attention_masks)
        }
