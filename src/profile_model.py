import torch
from torch.profiler import profile, record_function, ProfilerActivity
from train import CompressionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CompressionModel(128, 64, 6, 32, 16).to(device)
inputs = torch.randn(8, 1, 28, 28).to(device)

activities = [ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
