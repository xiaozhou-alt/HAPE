import torch
from torch import nn
import torch.nn.functional as F

#    "ViT-B-16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",

def convert_state_dict(state_dict: dict):
    print('clip_vit_b_16 state_dict_mapping')

    key_mapping = {
    }

    auto_mappings = {
        k: k.replace("visual.", "", 1)
        for k in state_dict.keys()
        if k.startswith("visual.") and k not in key_mapping
    }

    full_mapping = {**auto_mappings, **key_mapping}

    new_state_dict = {}
    for src_key, tgt_key in full_mapping.items():
        if src_key in state_dict:
            new_state_dict[tgt_key] = state_dict[src_key]

    return new_state_dict
