import os
import torch
import numpy as np
from transformers import (
    BartConfig, BartTokenizer, BartForConditionalGeneration,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import json
import re

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 27

token_vocab_file = 'PLM_TOKEN_DIR/vocab.json'
token_merge_file = 'PLM_TOKEN_DIR/merge.txt'

pretrained_model_dir = '/OUTPUT_DIR/'
train_json_dir = "PLM_TRAIN_DIR"
val_json = "./PLM_test.json"

tokenizer = BartTokenizer(token_vocab_file,token_merge_file)
config = BartConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=12,
    decoder_attention_heads=12,
    decoder_ffn_dim=3072,
    activation_function='gelu',
    dropout=0.1,
    attention_dropout=0.1,
    max_position_embeddings=1024,
    init_std=0.02,
    scale_embedding=False
)
model = BartForConditionalGeneration(config)

config = BartConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=768,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=12,
    decoder_attention_heads=12,
    decoder_ffn_dim=3072,
    activation_function='gelu',
    dropout=0.1,
    attention_dropout=0.1,
    max_position_embeddings=1024,
    init_std=0.02,
    scale_embedding=False
)

max_token_len = 27

class WordEmbeddingDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len=27):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        self.filtered_data = []
        for item in data:
            word = item["word"] if isinstance(item, dict) and "word" in item else str(item)
            tokenized = self.tokenizer.encode(word, add_special_tokens=False)
            if len(tokenized) <= self.max_token_len:
                self.filtered_data.append(word)

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        word = self.filtered_data[idx]
        input_ids = self.tokenizer(
            word,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
            add_special_tokens=False
        )["input_ids"].squeeze(0).to(dtype=torch.long)

        return {"input_ids": input_ids, "text": word}

class ProcessedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BARTTextInfillingCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, mask_token="<mask>"):
        super().__init__(tokenizer)
        self.mask_token = mask_token
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)

    def text_infilling(self, text, start_idx, num_to_mask=1):
        num_chars = len(text)
        if num_chars == 0:
            return text

        masked_indices = set()
        max_possible_length = num_chars - start_idx
        if max_possible_length <= 0:
            return text

        span_length = max(1, int(random.uniform(0.2, 0.4) * max_possible_length))  
        for i in range(start_idx, min(start_idx + span_length, num_chars)):
            if re.match(r'[A-Za-z]', text[i]):
                masked_indices.add(i)

        if not masked_indices:
            return text

        output_chars = []
        for i in range(num_chars):
            if i in masked_indices:
                output_chars.append(self.mask_token)
            else:
                output_chars.append(text[i])
        return "".join(output_chars)

    def __call__(self, examples):
        original_texts = [
            self.tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
            for ex in examples if ex["input_ids"] is not None
        ]

        input_texts = []
        label_texts = []

        for text in original_texts:
            if not text.strip():
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_token_len:
                for each_char_idx in range(len(text) - 1):
                    corrupted = self.text_infilling(text, each_char_idx, num_to_mask=1)
                    corrupted_ids = self.tokenizer.encode(corrupted, add_special_tokens=False)

                    if len(corrupted_ids) == len(token_ids):  
                        input_texts.append(corrupted)
                        label_texts.append(text)

        if not input_texts:
            input_texts = [""]
            label_texts = [""]

        model_inputs = self.tokenizer(
            input_texts,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
            max_length=max_token_len,
            add_special_tokens=False
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                label_texts,
                truncation=True,
                return_tensors="pt",
                padding="max_length",
                max_length=max_token_len,
                add_special_tokens=False
            )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs

entire_train_set = []
for each_train_json in os.listdir(train_json_dir):
    each_train_json_path = os.path.join(train_json_dir,each_train_json)
    each_f = open(each_train_json_path,'r')
    each_word_dict = json.load(each_f)
    entire_train_set += each_word_dict

entire_train_dict = list(set(entire_train_set))

flattened = [entire_train_dict[i] for i in range(len(entire_train_dict))]
train_dataset = WordEmbeddingDataset(flattened,tokenizer)

each_val = open(val_json,'r')
val_dict  = json.load(each_val)
flattened_val = [val_dict[i] for i in range(len(val_dict))]
val_dataset = WordEmbeddingDataset(flattened_val,tokenizer)

infilling_collator = BARTTextInfillingCollator(
    tokenizer=tokenizer,
    mask_token="<mask>"
)

model.to(device)

print('Number of parameters:', model.num_parameters())

# Training Arguments
training_args = TrainingArguments(
    output_dir=pretrained_model_dir,
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir=pretrained_model_dir,
    logging_strategy="epoch",
    fp16=True,
    save_safetensors=False,
    max_grad_norm=1.0,
)

# Metric computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    correct = (predictions == labels)
    correct = np.array(correct)
    word_correct = np.all(correct, axis=1, keepdims=True)
    accuracy = word_correct.sum() / word_correct.shape[0]
    print("=" * 50)
    print(f"Validation Results: MLM Accuracy: {accuracy:.4f}")
    print("=" * 50)
    return {"mlm_accuracy": accuracy}

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=infilling_collator,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits[0], dim=-1),
)


trainer.train()
trainer.save_model(pretrained_model_dir)
