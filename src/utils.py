import numpy as np
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM




def get_llm(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    try:
        model.seqlen = model.config.max_position_embeddings
    except:
        pass
    return model



# Function to select the appropriate loader based on dataset name
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False
):

    if name in ["alpaca", "alpaca_cleaned", "alpaca_cleaned_no_safety"]:
        return get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle, dataset=name)
    else:
        return get_custom(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, tag=name
        )


def get_custom(nsamples, seed, seqlen, tokenizer, disentangle=False, tag="filtered_bias"):
    # Load train and test datasets
    data_files = {"train": f"data/{tag}.csv"}

    traindata = load_dataset("csv", data_files=data_files, split="train")
    print('len(traindata)', len(traindata))  
    print('seqlen', seqlen)  
    print('nsamples', nsamples)  

    trainloader = []
    random.seed(seed)
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        print('trainenc.input_ids', trainenc.input_ids)

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None



def get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="alpaca"):
    if dataset == "alpaca":
        data_files = {"train": "./data/alpaca_train.csv"}
    elif dataset == "alpaca_cleaned":
        data_files = {"train": "./data/alpaca_cleaned_train.csv"}
    elif dataset == "alpaca_cleaned_no_safety":
        data_files = {"train": "./data/alpaca_cleaned_no_safety_train.csv"}
    else:
        raise ValueError("Dataset not supported")
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None