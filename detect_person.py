from PIL import Image
from locateanything_worker import LocateAnythingWorker

print("Loading model...")

worker = LocateAnythingWorker(
    "nvidia/LocateAnything-3B",
    device="cuda"
)

print("Loading image...")

img = Image.open("test.png").convert("RGB")

print("Running detection...")

result = worker.detect(
    img,
    ["person"]
)

print("\nMODEL OUTPUT:\n")
print(result["answer"])