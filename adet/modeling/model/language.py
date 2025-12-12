from transformers import BertTokenizer, BertForMaskedLM, BartTokenizer, BartForConditionalGeneration,BartConfig
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartDecoder

init_BART_dec_path = 'PLM_PATH/bart_decoder_lm_head.pt'
init_BART_path = 'PLM_PATH/checkpoint/'

ckpt_BART = ''
class DecoderWithLMHead(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config 
        self.decoder = BartDecoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embed_tokens.weight  
        self.encoder_proj = None  # will be initialized in forward()

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, decoder_position_ids=None):
        if self.encoder_proj is None:
            input_dim = encoder_hidden_states.size(-1)
            if input_dim != self.config.d_model:
                self.encoder_proj = nn.Linear(input_dim, self.config.d_model).to(encoder_hidden_states.device)
            else:
                self.encoder_proj = nn.Identity()
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)

        decoder_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            # attention_mask=attention_mask,
        )
       
        logits = self.lm_head(decoder_out.last_hidden_state)
        return logits

def get_checkpoint_BART(ckpt_from_spotter):
    global ckpt_BART
    ckpt_BART = ckpt_from_spotter

def load_checkpoint_BART(ckpt):
    bart_state_dict = {}
    if ckpt != '' and not ckpt.endswith('.pkl'):
        full_ckpt = torch.load(ckpt)
        bart_state_dict = {
                k.replace("criterion.lang_model.", ""): v
                for k, v in full_ckpt["model"].items()
                if k.startswith("criterion.lang_model.")
            }
    return bart_state_dict

def get_language_model_BART_dec():

    config = BartConfig(
    vocab_size=194,
    d_model=768,
    decoder_attention_heads=12,
    decoder_ffn_dim=3072,
    decoder_layers=6,
    max_position_embeddings=1024)

    decoder_model = DecoderWithLMHead(config)

    print(f'=========== Uploading decoder weights from {init_BART_dec_path} ===========')
    state_dict = torch.load(init_BART_dec_path)

    decoder_state = {f"decoder.{k}": v for k, v in state_dict.items() if not k.startswith("lm_head")}
    lm_head_state = {k: v for k, v in state_dict.items() if k.startswith("lm_head")}
    full_state = {**decoder_state, **lm_head_state}
    decoder_model.load_state_dict(full_state, strict=False)

    global ckpt_BART
    ckpt_lang_dict = load_checkpoint_BART(ckpt_BART)
    if len(ckpt_lang_dict) != 0:
        print(f'=========== Uploading LM weights from {ckpt_BART} ===========')
        decoder_model.load_state_dict(ckpt_lang_dict)
        # decoder_model.load_state_dict(ckpt_lang_dict, strict=False)
    for param in decoder_model.parameters():
        param.requires_grad = True


    torch.cuda.empty_cache()
    return decoder_model

def get_language_tokenizer_BART():
    # can use the same tokenizer during training
    tokenizer = BartTokenizer.from_pretrained(init_BART_path)
    return tokenizer