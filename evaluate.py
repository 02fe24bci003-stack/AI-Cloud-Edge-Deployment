
import requests
import matplotlib.pyplot as plt
import os
import time

# Docker Edge and Render Cloud URLs
urls = {
    "Docker Edge": "http://host.docker.internal:8000/predict",
    "Render Cloud": "https://ai-cloud-edge-deployment.onrender.com/predict"
}

# Test sentences and expected sentiments
test_data = [
    ("I love this product", "Positive"),
    ("This is amazing", "Positive"),
    ("I hate this product", "Negative"),
    ("This is very bad", "Negative"),
    ("The product is okay", "Neutral"),
    ("It is average", "Neutral")
]

names = []
accuracies = []
latencies = []
memories = []

for name, url in urls.items():
    print("\nEvaluating:", name)

    correct = 0
    total_time = 0
    total_memory = 0

    for sentence, actual in test_data:
        start = time.perf_counter()

        response = requests.post(
            url,
            json={"text": sentence},
            timeout=60
        )

        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()

        result = response.json()

        if result["predicted_sentiment"] == actual:
            correct += 1

        total_time += elapsed
        total_memory += result["memory_usage_mb"]

    accuracy = (correct / len(test_data)) * 100
    latency = total_time / len(test_data)
    memory = total_memory / len(test_data)

    names.append(name)
    accuracies.append(accuracy)
    latencies.append(latency)
    memories.append(memory)

    print("Accuracy:", accuracy)
    print("Average latency:", latency, "ms")
    print("Average memory:", memory, "MB")

# Create output folder
os.makedirs("/output", exist_ok=True)

# Accuracy graph
plt.figure()
plt.bar(names, accuracies)
plt.title("Cloud vs Edge Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig("/output/accuracy_graph.png")
plt.close()

# Latency graph
plt.figure()
plt.bar(names, latencies)
plt.title("Cloud vs Edge Latency")
plt.ylabel("Average Latency (ms)")
plt.savefig("/output/latency_graph.png")
plt.close()

# Memory graph
plt.figure()
plt.bar(names, memories)
plt.title("Cloud vs Edge Memory Usage")
plt.ylabel("Average Memory (MB)")
plt.savefig("/output/memory_graph.png")
plt.close()

print("\nAll three graphs generated successfully!")
