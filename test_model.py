from locateanything_worker import LocateAnythingWorker

print("Loading LocateAnything model...")

worker = LocateAnythingWorker(
    "nvidia/LocateAnything-3B",
    device="cuda"
)

print("Model loaded successfully!")