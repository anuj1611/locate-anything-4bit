import torch
from transformers import BitsAndBytesConfig
from PIL import Image, ImageDraw
from locateanything_worker import LocateAnythingWorker

print("Configuring 4-bit quantization...")
# 4-bit config tailored for GTX 1650
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # GTX 1650 lacks bfloat16 hardware acceleration, float16 is optimal
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
worker = LocateAnythingWorker(
    "nvidia/LocateAnything-3B",
    device="cuda",
    dtype=torch.float16,
    quantization_config=quantization_config
)

print("Loading image...")
img = Image.open("test.png").convert("RGB")

print("Running detection...")
result = worker.detect(
    img,
    ["person"]
)

print("\nMODEL OUTPUT:\n")
answer = result["answer"]
print(answer)

# Parse boxes using the built-in parser
w, h = img.size
boxes = LocateAnythingWorker.parse_boxes(answer, w, h)

print(f"\nFound {len(boxes)} boxes. Drawing...")
draw = ImageDraw.Draw(img)
for box in boxes:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    # Draw a 3-pixel thick red rectangle
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

# Save and optionally display the result
output_file = "output_test.png"
img.save(output_file)
print(f"Saved visual output to {output_file}")
try:
    img.show()
except Exception as e:
    print("Could not automatically display the image window, open", output_file, "manually.")
