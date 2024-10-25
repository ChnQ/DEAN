import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM

from dean import deactivating_by_dean
from eval_ppl import eval_ppl
from utils import get_llm
from eval_salad import eval_fairness_and_privacy_mdjudge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument(
        "--dataset1",
        type=str,
        default="fairness",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default="privacy",
    )
    parser.add_argument(
        "--target_module",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=5e-7,
    )
    parser.add_argument(
        "--q",
        type=float,
        default=5e-7,
    )

    args = parser.parse_args()
    print(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model = get_llm(args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )


    deactivating_by_dean(args=args, model=model,)

    # ================= ppl test ====================
    ppl_test = eval_ppl(args, model, tokenizer)
    print(f"wikitext perplexity {ppl_test}")


    # ================== save the model after applying DEAN =====================
    dean_path = os.path.join(
        './',
        f"{args.model_path.split('/')[-1]}_target_module_{args.target_module}_ratio_{args.p}",
    )
    model.save_pretrained(dean_path)
    tokenizer.save_pretrained(dean_path)


    trust_results = {}

    vllm_model = LLM(model=dean_path, gpu_memory_utilization=0.6)

    eval_model = AutoModelForCausalLM.from_pretrained("OpenSafetyLab/MD-Judge-v0.1").to("cuda")

    ## ================== eval fairness ==================

    trust_results['fairness'] = eval_fairness_and_privacy_mdjudge(
        category='fairness',
        vllm_model=vllm_model,
        eval_model=eval_model
    )

    ## ================== eval privacy ==================
    trust_results['privacy'] = eval_fairness_and_privacy_mdjudge(
        category='privacy',
        vllm_model=vllm_model,
        eval_model=eval_model
    )

    print('===================== params =====================')
    print(f'[model] {args.model}')
    print(f'[target_module] {args.target_module}')
    print(f'[ratio_p] {args.p}')
    print(f'[ratio_q] {args.q}')
    print('===================== trustworthy results here =====================')
    print(trust_results)


if __name__ == "__main__":
    main()
