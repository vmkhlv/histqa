import torch
import torch.nn.functional as F
from .utils import tokenizer, singleton
from transformers import BertConfig, BertForTokenClassification, BertTokenizer


@singleton
class HistNER:
    def __init__(self, model_dir, max_seq_len=128):
        self.model_dir = model_dir
        self.max_seq_length = max_seq_len
        self.config = BertConfig.from_pretrained(self.model_dir)
        self.label_map = self.config.id2label
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.model_dir, do_lower_case=False, config=self.config
        )
        self.model = BertForTokenClassification.from_pretrained(
            self.model_dir, config=self.config
        )
        self.model.eval()

    def convert_to_features(
        self, text, pad_token=0, cls_token_segment_id=1, tokenizer=tokenizer
    ):
        tokenized_text = tokenizer.tokenize(text)
        tokens, valid_positions = [self.bert_tokenizer.cls_token], []
        for i, token in enumerate(tokenized_text):
            token = self.bert_tokenizer.tokenize(token)
            tokens.extend(token)
            for j in range(len(token)):
                valid_positions.append(1) if j == 0 else valid_positions.append(0)
        tokens.append(self.bert_tokenizer.sep_token)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [pad_token] * len(tokens)
        input_mask = [cls_token_segment_id] * len(input_ids)

        if len(input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            segment_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length

        return tokenized_text, input_ids, input_mask, segment_ids, valid_positions

    def predict(self, text):
        """
        :param text: str
        :return: list of tuples (token, label)
        """
        (
            tokens,
            input_ids,
            input_mask,
            segment_ids,
            valid_positions,
        ) = self.convert_to_features(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_mask = torch.tensor([input_mask], dtype=torch.long)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)
        logits = F.softmax(logits[0], dim=2)
        logits_label = torch.argmax(logits, dim=2)
        logits_label = logits_label.detach().cpu().numpy()
        logits_confidence = [
            values[label].item() for values, label in zip(logits[0], logits_label[0])
        ]

        logits_label = [
            logits_label[0][index]
            for index, i in enumerate(input_mask[0])
            if i.item() == 1
        ]
        logits_label.pop(0)
        logits_label.pop()

        labels = []
        for valid, label in zip(valid_positions, logits_label):
            if valid:
                labels.append(self.label_map[label])
        prediction = [
            (token, label)
            for token, label, confidence in zip(tokens, labels, logits_confidence)
        ]

        return prediction
