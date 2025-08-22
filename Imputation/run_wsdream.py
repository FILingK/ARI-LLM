import os
import subprocess
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seq_len = "550"
model = "TM-LLM"
percent = 100
mask_rate = "0.1"
train_epochs ="2"
sample_num = 1000 #number of training samples
llm_model = "gpt2"
Lambda = 2
itr = "1"
command = [
    "/root/miniconda3/bin/python", "run.py",
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", r"../datasets/net_traffic/wsdream",
    "--data_path", "wsdream.csv",
    "--model_id", f"wsdream_1_sample_rate_{mask_rate}_{model}_samplenum{sample_num}",
    "--sample_num", str(sample_num),
    "--llm_model", llm_model,
    "--data", "net_traffic_trans",
    "--seq_len", seq_len,
    "--batch_size", "20",
    "--learning_rate", "0.001",
    '--mlp', "1",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "768",
    "--enc_in", "64",     # input of enc_embedding && output of flatten head
    "--dec_in", seq_len,     # feature
    "--c_out", seq_len,       # mlp
    "--Lambda", str(Lambda),
    "--freq", "h",
    "--percent", str(percent),
    "--gpt_layer", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate
]

subprocess.run(command)
