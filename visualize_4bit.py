"""
visualize_4bit.py
-----------------
Draws the bounding boxes from LocateAnything-3B (4-bit) output onto the image
and saves a result PNG. Run this AFTER run_4bit.py succeeds.

Usage:
    .\\venv\\Scripts\\python.exe visualize_4bit.py
"""

import re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "nvidia/LocateAnything-3B"
IMAGE_PATH = "test.png"
OUTPUT     = "result_4bit.png"
QUESTION   = "Locate all the instances that matches the following description: person."
MAX_SIDE   = 448   # resize shorter side to this before inference

# ── 4-bit config ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading tokenizer and processor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading model in 4-bit...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
torch.cuda.empty_cache()
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# ── Resize image ──────────────────────────────────────────────────────────────
img_raw = Image.open(IMAGE_PATH).convert("RGB")
w_orig, h_orig = img_raw.size
scale = MAX_SIDE / min(w_orig, h_orig)
if scale < 1.0:
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    img = img_raw.resize((new_w, new_h), Image.LANCZOS)
    print(f"Image resized: {w_orig}x{h_orig} -> {new_w}x{new_h}")
else:
    img = img_raw
    new_w, new_h = w_orig, h_orig

# ── Inference ─────────────────────────────────────────────────────────────────
print("Running inference...")
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text",  "text": QUESTION},
]}]

text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images_proc, videos_proc = processor.process_vision_info(messages)
inputs = processor(text=[text], images=images_proc, videos=videos_proc, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

with torch.no_grad():
    response = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=tokenizer,
        max_new_tokens=512,
        use_cache=True,
        generation_mode="slow",
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.1,
        verbose=False,
    )

answer = response[0] if isinstance(response, tuple) else response
print(f"\nRaw output:\n{answer}\n")

# ── Parse boxes ───────────────────────────────────────────────────────────────
# Output format: <ref>label</ref><box><x1><y1><x2><y2></box>
# Coordinates are integers in [0, 1000] (relative * 1000)
def parse_output(answer, img_w, img_h):
    results = []
    # Match label + boxes together
    for block in re.finditer(r"<ref>(.*?)</ref>((?:<box>.*?</box>)+)", answer):
        label = block.group(1)
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", block.group(2)):
            x1, y1, x2, y2 = [int(g) / 1000 for g in m.groups()]
            results.append({
                "label": label,
                "x1": x1 * img_w, "y1": y1 * img_h,
                "x2": x2 * img_w, "y2": y2 * img_h,
            })
    return results

boxes = parse_output(answer, new_w, new_h)
print(f"Detected {len(boxes)} instances")

# ── Draw on image ─────────────────────────────────────────────────────────────
draw_img = img.copy()
draw = ImageDraw.Draw(draw_img)

colors = ["#FF4444", "#44FF44", "#4488FF", "#FFAA00", "#FF44FF", "#00FFFF"]
try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

for i, box in enumerate(boxes):
    color = colors[i % len(colors)]
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    label_text = f"{box['label']} {i+1}"
    draw.text((x1 + 2, y1 + 2), label_text, fill=color, font=font)

draw_img.save(OUTPUT)
print(f"\nSaved: {OUTPUT}  ({new_w}x{new_h}, {len(boxes)} boxes drawn)")
print("Open result_4bit.png to inspect the detections.")
