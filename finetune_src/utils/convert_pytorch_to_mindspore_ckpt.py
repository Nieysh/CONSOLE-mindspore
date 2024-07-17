import torch

model_path = "../trained_models/vitbase-6tasks-pretrain/model_step_130000.pt"
ckpt_weights = torch.load(model_path)

print(ckpt_weights.keys())