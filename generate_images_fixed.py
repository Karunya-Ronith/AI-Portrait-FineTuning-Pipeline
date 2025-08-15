import os
import torch
import argparse
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_trained_pipeline(model_dir, base_model="runwayml/stable-diffusion-v1-5"):
    """Load the trained LoRA model and create a pipeline"""
    print("Loading base model components...")
    
    # Load base model components
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    
    # Load LoRA weights
    lora_path = os.path.join(model_dir, "lora")
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}")
        unet = PeftModel.from_pretrained(unet, lora_path)
        print("✓ LoRA weights loaded successfully")
    else:
        print(f"⚠ LoRA weights not found at {lora_path}")
        return None
    
    # Create pipeline
    pipeline = StableDiffusionPipeline(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        scheduler=scheduler,
        feature_extractor=None,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
        print("✓ Pipeline loaded on GPU")
    else:
        print("⚠ GPU not available, using CPU (will be slow)")
    
    return pipeline

def generate_image(pipeline, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5, seed=None):
    """Generate a single image"""
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None
    
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Generate images using trained Leonardo DiCaprio LoRA model")
    parser.add_argument("--model_dir", default="models", help="Directory containing trained model")
    parser.add_argument("--output_dir", default="outputs", help="Output directory for generated images")
    parser.add_argument("--prompt", default="leonardo dicaprio eating", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="Guidance scale")
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility")
    parser.add_argument("--width", default=512, type=int, help="Image width")
    parser.add_argument("--height", default=512, type=int, help="Image height")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pipeline
    pipeline = load_trained_pipeline(args.model_dir)
    if pipeline is None:
        print("❌ Failed to load trained model. Please check if training completed successfully.")
        return
    
    # Set image dimensions
    pipeline.width = args.width
    pipeline.height = args.height
    
    print(f"\n🎨 Generating {args.num_images} image(s) with prompt: '{args.prompt}'")
    
    # Generate images
    for i in tqdm(range(args.num_images), desc="Generating images"):
        current_seed = args.seed if args.seed is not None else torch.randint(0, 1000000, (1,)).item()
        
        try:
            image = generate_image(
                pipeline=pipeline,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=current_seed
            )
            
            # Save image
            filename = f"dicaprio_{i+1:03d}_seed_{current_seed}.png"
            filepath = os.path.join(args.output_dir, filename)
            image.save(filepath)
            print(f"✓ Saved: {filename}")
            
        except Exception as e:
            print(f"✗ Error generating image {i+1}: {e}")
    
    print(f"\n🎉 Generation complete! {args.num_images} image(s) saved to {args.output_dir}")
    print(f"\n💡 Try more prompts:")
    print(f"   python generate_images_fixed.py --prompt 'leonardo dicaprio portrait'")
    print(f"   python generate_images_fixed.py --prompt 'leonardo dicaprio on a red carpet'")
    print(f"   python generate_images_fixed.py --prompt 'leonardo dicaprio with sunglasses'")

if __name__ == "__main__":
    main() 