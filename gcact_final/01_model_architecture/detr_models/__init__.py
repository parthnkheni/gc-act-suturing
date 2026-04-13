# __init__.py  -- Model Registry
# Simple file that exposes the model build functions so other files can import
# them. When code says "build_ACT_model", it routes here and then to detr_vae.py.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp


def build_ACT_model(args):
    return build_vae(args)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)
