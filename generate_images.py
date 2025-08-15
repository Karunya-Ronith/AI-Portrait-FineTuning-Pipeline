import os
import torch
import argparse
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_trained_pipeline(model_dir, base_model="runwayml/stable-diffusion-v1-5"):
    """Load the trained pipeline with LoRA weights"""
    pipeline_path = os.path.join(model_dir, "pipeline")
    
    if os.path.exists(pipeline_path):
        print(f"Loading trained pipeline from {pipeline_path}")
        pipeline = StableDiffusionPipeline.from_pretrained(pipeline_path)
    else:
        print(f"Trained pipeline not found at {pipeline_path}")
        print("Please run training first: python train_lora_simple.py")
        return None
    
    # Move to GPU if available
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
    
    # Generate image
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
        return
    
    # Set image dimensions
    pipeline.width = args.width
    pipeline.height = args.height
    
    print(f"\nGenerating {args.num_images} image(s) with prompt: '{args.prompt}'")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    
    # Generate images
    for i in tqdm(range(args.num_images), desc="Generating images"):
        # Use different seed for each image if not specified
        current_seed = args.seed if args.seed is not None else torch.randint(0, 1000000, (1,)).item()
        
        # Generate image
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
        
        print(f"✓ Generated: {filename}")
    
    print(f"\n🎉 Generation complete! {args.num_images} image(s) saved to {args.output_dir}")
    
    # Show some example prompts
    print("\n💡 Try these example prompts:")
    print("  - 'leonardo dicaprio eating'")
    print("  - 'leonardo dicaprio smiling'")
    print("  - 'leonardo dicaprio in a suit'")
    print("  - 'leonardo dicaprio portrait'")
    print("  - 'leonardo dicaprio on a red carpet'")

if __name__ == "__main__":
    main() 