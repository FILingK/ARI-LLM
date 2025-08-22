import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from layers.Embed import DataEmbedding, DataEmbedding_wo_time
from peft import LoraConfig, get_peft_model


class FlattenHead(nn.Module):
    def __init__(self, d_model, Tstep):
        super().__init__()
        self.Tstep = Tstep
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(384, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, self.Tstep),
        )

    def forward(self, x):
        B, N, d_model = x.shape
        x = x.reshape(B * N, d_model)

        x = self.mlp(x)
        x = x.reshape(B, N, self.Tstep)
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff

        self.configs = configs

        self.flattenhead = FlattenHead(configs.d_model, self.configs.enc_in)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq,
                                           configs.dropout)  # modify***
        if configs.llm_model == 'gpt2':
            self.llm_model = GPT2Model.from_pretrained('../GPT2'
                                                       , output_attentions=True, output_hidden_states=True)
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
            self.tokenizer = AutoTokenizer.from_pretrained(
                '../GPT2',
                trust_remote_code=True,
                local_files_only=True
            )
        elif configs.llm_model == 'deepseek_R1':
            # Load model directly
            self.llm_config = AutoConfig.from_pretrained('../deepseek_R1_1.5b')
            self.llm_config.num_hidden_layers = configs.gpt_layers
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                '../deepseek_R1_1.5b',
                trust_remote_code=True,
                local_files_only=True,  # If the model file exists locally, load it from the local storage.
                config=self.llm_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                '../deepseek_R1_1.5b',
                trust_remote_code=True,
                local_files_only=True
            )
        elif configs.llm_model == 'Llama_3':
            # Load model directly
            self.llm_config = AutoConfig.from_pretrained('../Llama')
            self.llm_config.num_hidden_layers = configs.gpt_layers
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                '../Llama',
                trust_remote_code=True,
                local_files_only=True,  # If the model file exists locally, load it from the local storage.
                config=self.llm_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                '../Llama',
                trust_remote_code=True,
                local_files_only=True
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for i, (name, param) in enumerate(self.llm_model.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.llm_model.to(device=device)

        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)


    def forward(self, x_enc, x_mark_enc, known_rate,target_rate):
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc,known_rate,target_rate)
            return dec_out  # [B, L, D]

        return None

    def imputation(self, x_enc, x_mark_enc,known_rate,target_rate):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,d_model]
        prompt = []
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            if self.configs.data_path == 'abilene.csv':
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"Dataset Specification:12 routers (v1-v12) with 144 OD flows (x1-x144)，Each flow xij represents traffic from vi to vj, indexed as (i-1)*12+j (i,j∈1..12)"
                    f"- Temporal sequence length: {self.seq_len} time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"During the training process, we select fixed-interval rows as sampled data. Missing values are initially imputed using the nearest sampled data."
                    f"Learn spatiotemporal patterns in traffic flows. Perform progressive matrix completion through exponential imputation:"
                    f"Stage: 2%→4%→8%→16%→32%→64%→100%"
                    f"Current Task:"
                    f"- Advance from {known_rate}% to {target_rate}% completion"
                    # f"Input statistics: "
                    # f"min value {min_values_str},max value {max_values_str}, median value {median_values_str}"
                )
            elif self.configs.data_path == 'wsdream.csv':
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"Dataset Specification:550 OD flows (x1-x550)"
                    f"- Temporal sequence length: {64} time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"During the training process, we select fixed-interval rows as sampled data. Missing values are initially imputed using the nearest sampled data."
                    f"Learn spatiotemporal patterns in traffic flows. Perform progressive matrix completion through exponential imputation:"
                    f"Stage: 2%→4%→8%→16%→32%→64%→100%"
                    f"Current Task:"
                    f"- Advance from {known_rate}% to {target_rate}% completion"
                    # f"Input statistics: "
                    # f"min value {min_values_str},max value {max_values_str}, median value {median_values_str}"
                )
            else:
                prompt_ = (
                    f"Network Traffic Matrix Completion Task"
                    f"Dataset Specification:23 routers (v1-v23) with 529 OD flows (x1-x529)，Each flow xij represents traffic from vi to vj, indexed as (i-1)*23+j (i,j∈1..23)"
                    f"- Temporal sequence length: {self.seq_len} time steps"
                    f"- The current traffic matrix has {known_rate}% sampling rate with significant missing data"
                    f"During the training process, we select fixed-interval rows as sampled data. Missing values are initially imputed using the nearest sampled data."
                    f"Learn spatiotemporal patterns in traffic flows. Perform progressive matrix completion through exponential imputation:"
                    f"Stage: 2%→4%→8%→16%→32%→64%→100%"
                    f"Current Task:"
                    f"- Advance from {known_rate}% to {target_rate}% completion"
                    # f"Input statistics: "
                    # f"min value {min_values_str},max value {max_values_str}, median value {median_values_str}"
                )
            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        combined_input = torch.cat((prompt_embeddings, enc_out), dim=1)  # Concatenate the prompt and the data
        if self.configs.llm_model == "gpt2":
            outputs = self.llm_model(inputs_embeds=combined_input).last_hidden_state


        else:
            outputs = self.llm_model(inputs_embeds=combined_input).hidden_states[-1]

        outputs = self.ln_proj(outputs[:, -self.configs.c_out:, :])
        dec_out = self.flattenhead(outputs).permute(0, 2, 1)

        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out
