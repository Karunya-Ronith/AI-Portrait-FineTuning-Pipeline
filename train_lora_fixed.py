import os
import torch
import argparse
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import Dataset
import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import math
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging

logger = get_logger(__name__)

def load_dataset(data_dir):
    """Load the prepared dataset"""
    images_dir = os.path.join(data_dir, "images")
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = [json.loads(line) for line in f]
    
    # Load images
    images = []
    texts = []
    
    for item in metadata:
        image_path = os.path.join(images_dir, item["file_name"])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            texts.append(item["text"])
    
    print(f"Loaded {len(images)} images with captions")
    
    return Dataset.from_dict({
        "image": images,
        "text": texts
    })

def collate_fn(examples):
    """Collate function for the dataset"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def main():
    parser = argparse.ArgumentParser(description="Train LoRA on Leonardo DiCaprio dataset")
    parser.add_argument("--dataset_dir", default="dataset", help="Path to prepared dataset")
    parser.add_argument("--output_dir", default="models", help="Output directory for trained model")
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5", help="Base model to fine-tune")
    parser.add_argument("--resolution", default=512, type=int, help="Image resolution")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int, help="Gradient accumulation steps")
    parser.add_argument("--max_train_steps", default=100, type=int, help="Maximum training steps")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--lora_rank", default=16, type=int, help="LoRA rank")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--mixed_precision", default="fp16", help="Mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
    
    # Load dataset
    dataset = load_dataset(args.dataset_dir)
    
    # Set up data transforms
    def transform_images(examples):
        images = [image.resize((args.resolution, args.resolution)) for image in examples["image"]]
        images = [np.array(image) for image in images]
        images = [image / 127.5 - 1.0 for image in images]
        return {"pixel_values": images}
    
    def tokenize_captions(examples):
        captions = [caption for caption in examples["text"]]
        text_inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {"input_ids": text_inputs.input_ids}
    
    # Apply transforms in correct order
    dataset = dataset.map(tokenize_captions, batched=True)
    dataset.set_transform(transform_images)
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Set up LoRA
    lora_attn_procs = {}
    for name, module in unet.named_modules():
        if hasattr(module, "to_k") and hasattr(module, "to_q") and hasattr(module, "to_v"):
            lora_attn_procs[f"{name}.to_k"] = LoRAAttnProcessor(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
            )
            lora_attn_procs[f"{name}.to_q"] = LoRAAttnProcessor(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
            )
            lora_attn_procs[f"{name}.to_v"] = LoRAAttnProcessor(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
            )
    
    # Set the LoRA layers
    unet.set_attn_processor(lora_attn_procs)
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_train_steps)
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    
    # Prepare everything with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move VAE and text encoder to device
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    
    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    
    global_step = 0
    
    for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        unet.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                
                # Backprop
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step >= args.max_train_steps:
                break
        
        # Log progress
        accelerator.log({"train_loss": total_loss.item() / len(train_dataloader)}, step=epoch)
        
        if global_step >= args.max_train_steps:
            break
    
    # Save the trained model
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    
    # Save LoRA weights
    lora_layers = AttnProcsLayers(unwrapped_unet.attn_processors)
    lora_layers.save_pretrained(os.path.join(args.output_dir, "lora"))
    
    # Save pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        unet=unwrapped_unet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
    )
    
    pipeline.save_pretrained(os.path.join(args.output_dir, "pipeline"))
    
    print(f"Training completed! Model saved to {args.output_dir}")
    print(f"LoRA weights saved to {os.path.join(args.output_dir, 'lora')}")
    print(f"Pipeline saved to {os.path.join(args.output_dir, 'pipeline')}")

if __name__ == "__main__":
    main() 