import torch
import torch.nn as nn
import re
import os
import pickle
import argparse
import numpy as np

from utils import get_llm
from utils import get_loaders
from functools import reduce
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = torch.zeros(
            [base.in_features], device=self.base.weight.device, requires_grad=False
        )
        self.n_samples = 0
        self.record_activation = True

    def clear_act_buffer(self):
        self.activation_norms.fill_(0.0)
        self.n_samples = 0

    def forward(self, x):

        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask]
            else:
                x_ = x

            bs = x_.nelement() // x_.shape[-1]
            self.activation_norms = self.activation_norms * (
                self.n_samples / (self.n_samples + bs)
            ) + (x_ * x_).view(-1, x_.shape[-1]).sum(dim=0) * (
                1.0 / (self.n_samples + bs)
            )
            self.n_samples += bs

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            # Extract the base nn.Linear module from ActLinear
            linear_module = module.base
            # Navigate to the parent module of the ActLinear module
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            print(f"Reverting {name}, parent: {parent_name}")
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            # Replace the ActLinear module with the extracted nn.Linear module
            setattr(parent_module, name.split(".")[-1], linear_module)

    return model


def make_Act(model, verbose=False):
    replace_map = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
                    # setattr(model, name_suffix, v)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


def compute_importance_score(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
):
    model = make_Act(model, verbose=False)

    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=True,
    )
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers
    saved_grad = {}
    for layer in range(num_hidden_layers):
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  

        model.zero_grad()
        model.requires_grad_(False)
        for name, module in model.named_modules():  
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("enabling grad for ", name)
                module.base.requires_grad_(True)
                saved_grad[name] = torch.zeros_like(
                    module.base.weight, device=module.base.weight.device
                )
                module.base.zero_grad()
            
        for batch in dataloader:
            inp, tar = batch[0].to(device), batch[1].to(device)
            model.zero_grad()
            with no_act_recording(model):
                loss = model(input_ids=inp, labels=tar)[0]
            loss.backward()
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    saved_grad[name] += module.base.weight.grad.abs()   

        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.base.weight.grad.copy_(saved_grad[name])
                saved_grad.pop(name)


        _compute_and_save(
            args,
            model,
            name_filter_fn=layer_filter_fn,
        )

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory


def _compute_and_save(
    args,
    model,
    name_filter_fn=None,
):
    dataset = args.dataset
    for name, module in model.named_modules():
        if name_filter_fn is not None and not name_filter_fn(name):
            continue
        if isinstance(module, ActLinear):
            i = re.search(r"\d+", name)
            if i:
                i = int(i.group())
            else:
                i = 0

            magnitude = torch.abs(module.base.weight.data)  #  weight W

            act = module.base.weight.grad.abs()  #  dW

            importance_score = magnitude * act  # importance score

            # save the importance score
            save_folder = os.path.join(
                f"significance_score/{args.nsamples}/{args.dataset}/{args.model.lower()}" 
            )  
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            target_file = os.path.join(
                save_folder, f"layer_{i}_name_{name}_weight.pkl"
            )
            with open(target_file, "wb") as f:
                print(
                    "Writing importance score in layer {} and name {} with {} to the file".format(
                        i, name, dataset
                    )
                )
                pickle.dump(importance_score, f)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()
    print(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )
    # Compute and save the importance score
    compute_importance_score(
        args=args,
        model=model,
        tokenizer=tokenizer
    )
