# ============================================================
# AI-BASED OBJECT REMOVAL & IMAGE RESTORATION SYSTEM
# ULTRA CINEMATIC FUTURISTIC UI VERSION
# ============================================================

# FEATURES
# ------------------------------------------------------------
# 1. Manual Mask Inpainting
# 2. Automatic AI Mask Generation
# 3. Stable Diffusion Inpainting
# 4. Ultra Dynamic Cinematic UI
# 5. Animated Background
# 6. Neon Glow Effects
# 7. Glassmorphism Design
# 8. Smooth Hover Animations
# ============================================================

# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================

!pip install -q diffusers transformers accelerate torch pillow gradio rembg onnxruntime opencv-python

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
from rembg import remove
import numpy as np
import cv2

# ============================================================
# LOAD STABLE DIFFUSION MODEL
# ============================================================

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

# ============================================================
# HIGH QUALITY PROMPTS
# ============================================================

prompt = """
highly realistic natural background,
photorealistic environment,
seamless continuation of surrounding textures,
ultra realistic scene reconstruction,
clean background,
natural lighting,
high quality,
detailed environment
"""

negative_prompt = """
text,
logo,
watermark,
letters,
symbols,
badge,
graphic,
poster,
artifacts,
blurry,
distorted,
duplicate objects,
low quality,
cartoon,
painting,
drawing,
unnatural patches,
human remains,
object remains
"""

# ============================================================
# MANUAL MASK INPAINTING FUNCTION
# ============================================================

def manual_mask_inpaint(input_image, mask_image):

    image = input_image.resize((512, 512))
    mask = mask_image.resize((512, 512))

    # ========================================================
    # MASK REFINEMENT
    # ========================================================

    mask_np = np.array(mask)

    kernel = np.ones((11,11), np.uint8)

    mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    mask_np = cv2.GaussianBlur(mask_np, (7,7), 0)

    refined_mask = Image.fromarray(mask_np)

    # ========================================================
    # INPAINTING
    # ========================================================

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=refined_mask,
        guidance_scale=14,
        num_inference_steps=60
    ).images[0]

    return refined_mask, result

# ============================================================
# AUTOMATIC MASK GENERATION FUNCTION
# ============================================================

def auto_mask_inpaint(input_image):

    image = input_image.resize((512, 512))

    # ========================================================
    # AUTOMATIC FOREGROUND DETECTION
    # ========================================================

    removed_bg = remove(
        image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240
    )

    removed_bg_np = np.array(removed_bg)

    alpha = removed_bg_np[:, :, 3]

    alpha = cv2.threshold(
        alpha,
        40,
        255,
        cv2.THRESH_BINARY
    )[1]

    # ========================================================
    # MASK REFINEMENT
    # ========================================================

    kernel = np.ones((9,9), np.uint8)

    alpha = cv2.dilate(alpha, kernel, iterations=2)

    alpha = cv2.morphologyEx(
        alpha,
        cv2.MORPH_CLOSE,
        kernel
    )

    alpha = cv2.GaussianBlur(alpha, (5,5), 0)

    mask = Image.fromarray(alpha).convert("RGB")

    # ========================================================
    # INPAINTING
    # ========================================================

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=14,
        num_inference_steps=60
    ).images[0]

    return mask, result

# ============================================================
# ULTRA DYNAMIC CUSTOM CSS
# ============================================================

custom_css = """

/* =========================================================
BACKGROUND ANIMATION
========================================================= */

body {

    background: linear-gradient(
        -45deg,
        #020617,
        #0f172a,
        #111827,
        #1e293b,
        #312e81
    );

    background-size: 400% 400%;

    animation: gradientBG 15s ease infinite;

    overflow-x: hidden;
}

/* Animated gradient */

@keyframes gradientBG {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

/* =========================================================
MAIN CONTAINER
========================================================= */

.gradio-container {

    font-family: 'Poppins', sans-serif;

    color: white;

    animation: fadeIn 1.5s ease-in-out;
}

/* =========================================================
FADE ANIMATION
========================================================= */

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

/* =========================================================
TITLE ANIMATION
========================================================= */

h1 {

    text-align: center;

    font-size: 54px !important;

    font-weight: 900 !important;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #a78bfa,
        #f472b6
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    animation: glowText 2s ease-in-out infinite alternate;

    margin-bottom: 10px !important;
}

@keyframes glowText {

    from {
        text-shadow: 0px 0px 10px #7c3aed;
    }

    to {
        text-shadow: 0px 0px 30px #60a5fa;
    }
}

/* =========================================================
SUBTEXT
========================================================= */

p {

    color: #d1d5db !important;

    font-size: 18px !important;

    text-align: center;
}

/* =========================================================
GLASSMORPHISM PANELS
========================================================= */

.gr-box {

    background: rgba(255,255,255,0.08) !important;

    border-radius: 24px !important;

    border: 1px solid rgba(255,255,255,0.1);

    backdrop-filter: blur(18px);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.37);

    transition: all 0.4s ease;
}

/* Hover animation */

.gr-box:hover {

    transform: translateY(-5px);

    box-shadow:
        0px 0px 30px rgba(124,58,237,0.5);
}

/* =========================================================
BUTTONS
========================================================= */

.gr-button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed,
        #ec4899
    ) !important;

    color: white !important;

    border-radius: 18px !important;

    border: none !important;

    font-size: 20px !important;

    font-weight: bold !important;

    transition: all 0.3s ease-in-out !important;

    box-shadow:
        0px 0px 20px rgba(124,58,237,0.5);
}

/* Hover glow */

.gr-button:hover {

    transform: scale(1.08);

    box-shadow:
        0px 0px 40px rgba(236,72,153,0.8);

    letter-spacing: 1px;
}

/* =========================================================
IMAGE PANELS
========================================================= */

img {

    border-radius: 18px !important;

    transition: all 0.4s ease;
}

/* Image hover effect */

img:hover {

    transform: scale(1.02);

    box-shadow:
        0px 0px 25px rgba(96,165,250,0.5);
}

/* =========================================================
TABS
========================================================= */

button[role="tab"] {

    font-size: 18px !important;

    font-weight: bold !important;

    border-radius: 14px !important;

    transition: all 0.3s ease;
}

/* Active tab */

button[role="tab"][aria-selected="true"] {

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    ) !important;

    color: white !important;

    box-shadow:
        0px 0px 20px rgba(124,58,237,0.6);
}

/* =========================================================
FOOTER REMOVE
========================================================= */

footer {

    visibility: hidden;
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {

    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: linear-gradient(
        #2563eb,
        #7c3aed
    );

    border-radius: 10px;
}

"""

# ============================================================
# CREATE UI
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown("""
    # AI-Based Object Removal & Image Restoration

    ### Powered by Stable Diffusion Inpainting
    """)

    # ========================================================
    # MANUAL MASK TAB
    # ========================================================

    with gr.Tab("🎨 Manual Mask Inpainting"):

        gr.Markdown("""
        Upload:
        - Original Image
        - Mask Image

        White Region = Object to Remove
        """)

        with gr.Row():

            input_image_manual = gr.Image(
                type="pil",
                label="📷 Original Image",
                height=350
            )

            mask_image_manual = gr.Image(
                type="pil",
                label="🎭 Mask Image",
                height=350
            )

        manual_button = gr.Button(
            "✨ Generate AI Output"
        )

        with gr.Row():

            refined_mask_manual = gr.Image(
                type="pil",
                label="🧠 Refined Mask",
                height=350
            )

            output_manual = gr.Image(
                type="pil",
                label="🖼️ Generated Output",
                height=350
            )

        manual_button.click(
            fn=manual_mask_inpaint,
            inputs=[
                input_image_manual,
                mask_image_manual
            ],
            outputs=[
                refined_mask_manual,
                output_manual
            ]
        )

    # ========================================================
    # AUTOMATIC MASK TAB
    # ========================================================

    with gr.Tab("🤖 Automatic AI Masking"):

        gr.Markdown("""
        Upload only Original Image.

        AI automatically detects foreground object
        and removes it intelligently.
        """)

        input_image_auto = gr.Image(
            type="pil",
            label="📷 Original Image",
            height=350
        )

        auto_button = gr.Button(
            "🚀 Generate Automatically"
        )

        with gr.Row():

            generated_mask_auto = gr.Image(
                type="pil",
                label="🧠 Auto Generated Mask",
                height=350
            )

            output_auto = gr.Image(
                type="pil",
                label="🖼️ Generated Output",
                height=350
            )

        auto_button.click(
            fn=auto_mask_inpaint,
            inputs=input_image_auto,
            outputs=[
                generated_mask_auto,
                output_auto
            ]
        )

    # ========================================================
    # FOOTER
    # ========================================================

    gr.Markdown("""
    ---
    ### Developed using:
    - Stable Diffusion Inpainting
    - Gradio UI
    - OpenCV
    - REMBG AI Segmentation
    """)

# ============================================================
# LAUNCH WEBSITE
# ============================================================

demo.launch(
    share=True,
    debug=True
)