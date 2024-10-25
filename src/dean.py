import torch
import pickle
import torch.nn as nn


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def get_set_difference_mask(p, q, W_metric1, W_metric2):
    top_p = int(
        p * W_metric1.shape[1] * W_metric1.shape[0]
    ) 
    top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])  

    top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
    top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
    unique_p = torch.unique(top_p_indices)
    unique_q = torch.unique(top_q_indices)

    mask_only_metric1 = ~torch.isin(unique_p, unique_q)  
    mask_only_metric2 = ~torch.isin(unique_q, unique_p)  
    mask_intersection = ~(torch.ones_like(unique_p).bool() & mask_only_metric1 & mask_only_metric2)

    return mask_only_metric1, mask_only_metric2, mask_intersection, unique_q


def deactivating_by_dean(model, args):
    model.config.use_cache = False
    layers = model.model.layers
    metric1 = "alpaca_cleaned_no_safety"

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if args.target_module == 'mlp':
                if 'self_attn' in name:
                    print(f'[Note] Skip layer {i} {name} due to target module {args.target_module}')
                    continue
            elif args.target_module == 'self_attn':
                if 'mlp' in name:
                    print(f'[Note] Skip layer {i} {name} due to target module {args.target_module}')
                    continue

            print(f"Handling layer {i} name {name}")




            try:
                W_general = pickle.load(
                    open(
                        f"significance_score/{args.nsamples}/{metric1}/{args.model.lower()}/layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                        "rb",
                    ),
                )
                W_dataset1 = pickle.load(
                    open(
                        f"significance_score/{args.nsamples}/{args.dataset1}/{args.model.lower()}/layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                        "rb",
                    )
                )
                W_dataset2 = pickle.load(
                    open(
                        f"significance_score/{args.nsamples}/{args.dataset2}/{args.model.lower()}/layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                        "rb",
                    )
                )
            except Exception as e:  
                print(e)            

            _, mask_only_trust1, _, unique_trust1_indices = get_set_difference_mask(args.p, args.q, W_general, W_dataset1)
            _, mask_only_trust2, _, unique_trust2_indices = get_set_difference_mask(args.p, args.q, W_general, W_dataset2)
            mask_trust_intersection = mask_only_trust1 & mask_only_trust2
            mask_only_trust1 &= ~mask_trust_intersection
            mask_only_trust2 &= ~mask_trust_intersection

            filtered_indices = unique_trust1_indices[mask_trust_intersection]

            W_mask = torch.zeros_like(subset[name].weight.data) == 1 
            W_mask.view(-1)[filtered_indices] = True
            subset[name].weight.data[W_mask] = 0  ## set weights to zero

