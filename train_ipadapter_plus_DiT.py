import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL

from ip_adapter.resampler import Resampler
from models import DiT  # Assuming this is where the Lumina-Next DiT model is defined

# Dataset
class LuminaDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.data = json.load(open(json_file))
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }

class IPAdapterLuminaNext(torch.nn.Module):
    def __init__(self, dit_model, image_proj_model, num_tokens):
        super().__init__()
        self.dit_model = dit_model
        self.image_proj_model = image_proj_model
        self.num_tokens = num_tokens

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        noise_pred = self.dit_model(noisy_latents, timesteps, encoder_hidden_states)
        return noise_pred

def main():
    parser = argparse.ArgumentParser(description="Training script for IP-Adapter-Plus with Lumina-Next-SFT.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--data_json_file", type=str, required=True)
    parser.add_argument("--data_root_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="lumina_ip_adapter")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_tokens", type=int, default=16)
    args = parser.parse_args()

    accelerator = Accelerator()

    # Load models
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    text_encoder = AutoModelForCausalLM.from_pretrained("google/gemma-2b").get_decoder().cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    dit_model = DiT()  # Initialize your Lumina-Next DiT model here
    image_encoder = AutoModelForCausalLM.from_pretrained(args.image_encoder_path)

    # Initialize IP-Adapter components
    image_proj_model = Resampler(
        dim=dit_model.config.hidden_size,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=dit_model.config.hidden_size,
        ff_mult=4
    )

    ip_adapter = IPAdapterLuminaNext(dit_model, image_proj_model, args.num_tokens)

    # Freeze parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(ip_adapter.parameters(), lr=args.learning_rate)

    # Dataset and DataLoader
    train_dataset = LuminaDataset(args.data_json_file, tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    # Prepare for distributed training
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    # Training loop
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                latents = vae.encode(batch["images"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
                noisy_latents = latents + noise

                # Get text and image embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device), output_hidden_states=True).hidden_states[-2]

                # Forward pass
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)

                # Compute loss
                loss = F.mse_loss(noise_pred, noise)

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        # Save checkpoint
        if accelerator.is_main_process:
            accelerator.save_state(f"{args.output_dir}/checkpoint-{epoch}")

if __name__ == "__main__":
    main()