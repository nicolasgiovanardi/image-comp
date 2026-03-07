import torch
import time
from train import CompressionModel


def measure_latency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    model = CompressionModel(128, 64, 6, 32, 16).to(device)
    model.eval()

    # Warm-up
    for _ in range(20):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    iterations = 100
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    avg_latency = (time.time() - start_time) / iterations * 1000
    print(f"Average Inference Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    measure_latency()
