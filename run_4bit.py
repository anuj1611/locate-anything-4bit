"""
run_4bit.py
-----------
Runs LocateAnything-3B with 4-bit NF4 quantization via bitsandbytes.
Designed for GTX 1650 (4 GB VRAM) on Windows.

Usage:
    .\\venv\\Scripts\\python.exe run_4bit.py
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

# ── 1. 4-bit quantization config ──────────────────────────────────────────────
# nf4 = NormalFloat4 (best quality for weights)
# double quant = quantize the quantization constants too (saves ~0.4 GB extra)
# compute dtype = float16 (GTX 1650 does NOT support bfloat16 natively)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # fp16 for Turing (GTX 1650)
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = "nvidia/LocateAnything-3B"
DEVICE   = "cuda"

# ── 2. Load tokenizer + processor ─────────────────────────────────────────────
print("[1/3] Loading tokenizer and processor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("      [OK] Done")

# ── 3. Load model in 4-bit ────────────────────────────────────────────────────
# NOTE: do NOT call .to(device) after this — bitsandbytes handles device placement.
print("[2/3] Loading model in 4-bit (this may take 1–2 min on first run)...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",          # lets accelerate decide (puts on GPU when possible)
    low_cpu_mem_usage=True,     # avoids creating full fp16 copy in CPU RAM
)
model.eval()
# Free any leftover CPU/GPU cache from loading before inference
torch.cuda.empty_cache()
print("      [OK] Model loaded")

# ── 4. Quick VRAM check ───────────────────────────────────────────────────────
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved  = torch.cuda.memory_reserved(0) / 1e9
print("")
print(f"      VRAM after load : {allocated:.2f} GB used / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total")
print(f"      VRAM free (approx): {torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved:.2f} GB\n")

# ── 5. Run a simple inference ─────────────────────────────────────────────────
print("[3/3] Running inference on test.png ...")

IMAGE_PATH = "test.png"
QUESTION   = "Locate all the instances that matches the following description: person."

# --- Image resize -----------------------------------------------------------
# Moon-ViT processes at NATIVE resolution, which is the main VRAM killer.
# 1919x1079 generates ~2M patch tokens; ViT attention = O(n^2) in memory.
# Cap shorter side to MAX_SIDE px. 448 leaves ~1.2 GB for activations.
# Increase to 672 if your run succeeds, for better detection accuracy.
MAX_SIDE = 448
img_raw = Image.open(IMAGE_PATH).convert("RGB")
w, h = img_raw.size
scale = MAX_SIDE / min(w, h)
if scale < 1.0:  # only downscale, never upscale
    new_w, new_h = int(w * scale), int(h * scale)
    img = img_raw.resize((new_w, new_h), Image.LANCZOS)
    print(f"      Image resized: {w}x{h} -> {new_w}x{new_h} (MAX_SIDE={MAX_SIDE})")
else:
    img = img_raw
    print(f"      Image kept at original size: {w}x{h}")

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": QUESTION},
    ]}
]

text   = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = processor.process_vision_info(messages)
inputs = processor(
    text=[text], images=images, videos=videos, return_tensors="pt"
).to(DEVICE)

# Cast pixel_values to fp16 (matches bnb_4bit_compute_dtype)
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
        # Use "slow" (NTP) mode — MTP/hybrid decodes boxes in parallel
        # which requires extra activations; NTP is more memory-efficient.
        generation_mode="slow",
        temperature=0.0,   # greedy for stability
        do_sample=False,
        repetition_penalty=1.1,
        verbose=True,
    )

# VRAM after inference
post_alloc = torch.cuda.memory_allocated(0) / 1e9
print(f"\n      VRAM after inference: {post_alloc:.2f} GB used")

answer = response[0] if isinstance(response, tuple) else response
print("\n--- Model Output ---")
print(answer)
print("--------------------")
