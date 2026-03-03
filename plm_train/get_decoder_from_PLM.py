from transformers import BartTokenizer, BartForConditionalGeneration,BartConfig
import torch
import torch.nn.functional as F
import torch.nn as nn

init_BART_path =  'PLM_WEIGHT_DIR'
saved_decoder_path = 'OUTPUT_PLM_DEC_DIR/bart_decoder_lm_head.pt'

def get_dec_from_BART(bart_pretrained,saved_dec_path):
    model = BartForConditionalGeneration.from_pretrained(init_BART_path)
    tokenizer = BartTokenizer.from_pretrained(init_BART_path)
    print(tokenizer.vocab_size)
    decoder_only = model.model.decoder
    dec_state_dict = decoder_only.state_dict()
    lm_head_state_dict = {'lm_head.' + k: v for k, v in model.lm_head.state_dict().items()}

    full_state_dict = {**dec_state_dict, **lm_head_state_dict}
    torch.save(full_state_dict, saved_dec_path)

get_dec_from_BART(init_BART_path,saved_decoder_path)
