import gc
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from accelerate import Accelerator
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
# from skimage import exposure
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import csv

image_size = 64

class PromptGenerator:
    """Maintains your original prompt generation logic"""
    def __init__(self):
        self.phase_descriptions = {
            'I': {
                'name': 'Inflammatory',
                'characteristics': [
                    'redness and swelling',
                    'early wound healing',
                    'acute inflammation',
                    'wound debridement phase'
                ]
            },
            'P': {
                'name': 'Proliferative',
                'characteristics': [
                    'granulation tissue formation',
                    'pink to red tissue bed',
                    'active wound healing',
                    'new tissue growth'
                ]
            },
            'R': {
                'name': 'Remodeling',
                'characteristics': [
                    'wound contraction',
                    'epithelialization',
                    'scar tissue formation',
                    'mature healing'
                ]
            }
        }
        
        self.modality_descriptions = {
            'rgb': 'visible light photograph of',
            'depth_map': 'depth mapping of',
            'thermal_map': 'thermal imaging of'
        }

    def generate_prompt(self, modality, phase):
        phase_info = self.phase_descriptions[phase]
        base_desc = f"{self.modality_descriptions[modality]} diabetic foot ulcer (DFU)"
        phase_desc = f"in {phase_info['name']} phase"
        characteristics = f"showing {', '.join(phase_info['characteristics'])}"
        prompt = f"Medical imaging: {base_desc} {phase_desc}, {characteristics}"
        negative_prompt = "blurry, distorted, unrealistic, non-medical, artistic, cartoon"
        return prompt, negative_prompt

class WoundDataset(Dataset):
    def __init__(self, data_dir, phase, modality, tokenizer, cache_dir=None):
        self.data_dir = data_dir
        self.phase = phase
        self.modality = modality
        self.tokenizer = tokenizer
        self.prompt_generator = PromptGenerator()
        self.target_size = 64
        
        self.image_paths = [f for f in os.listdir(data_dir) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Generate prompt once since it's the same for all images
        self.prompt, _ = self.prompt_generator.generate_prompt(modality, phase)
        
        # Pre-encode prompt
        self.encoded_prompt = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        
        # Try to load cached tensors if they exist
        cache_path = f"{cache_dir or data_dir}/cached_{modality}_{phase}.pt"
        if os.path.exists(cache_path):
            print(f"Loading cached tensors from {cache_path}")
            self.cached_tensors = torch.load(cache_path)
        else:
            print("Processing and caching images...")
            self.cached_tensors = self.preprocess_images()
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(self.cached_tensors, cache_path)
                print(f"Cached tensors saved to {cache_path}")

    def preprocess_images(self):
        """Process all images once and store as tensors"""
        processed_tensors = []
        
        for img_path in self.image_paths:
            # Load and convert image
            image = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
            
            # Convert to tensor
            tensor = transforms.functional.to_tensor(image)
            
            # Resize keeping aspect ratio
            tensor = transforms.functional.resize(tensor, self.target_size, antialias=True)
            
            # Calculate padding
            h, w = tensor.shape[-2:]
            padding_h = max(0, self.target_size - h)
            padding_w = max(0, self.target_size - w)
            padding_h_top = padding_h // 2
            padding_h_bottom = padding_h - padding_h_top
            padding_w_left = padding_w // 2
            padding_w_right = padding_w - padding_w_left
            
            # Apply padding
            tensor = transforms.functional.pad(
                tensor,
                [padding_w_left, padding_h_top, padding_w_right, padding_h_bottom],
                fill=0
            )
            
            # Normalize
            tensor = transforms.functional.normalize(tensor, [0.5], [0.5])
            
            processed_tensors.append(tensor)
        
        return torch.stack(processed_tensors)

    def __len__(self):
        return len(self.cached_tensors)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.cached_tensors[idx],
            "input_ids": self.encoded_prompt
        }

def create_pinned_datasets(data_dir, phase, modality, tokenizer, train_ratio=0.8, 
                          pin_memory=True, cache_dir=None):
    """Create train and validation datasets with optional pinned memory"""
    # Create dataset
    dataset = WoundDataset(data_dir, phase, modality, tokenizer, cache_dir)
    
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    if pin_memory and torch.cuda.is_available():
        # Pin memory for faster GPU transfer
        dataset.cached_tensors = dataset.cached_tensors.pin_memory()
        dataset.encoded_prompt = dataset.encoded_prompt.pin_memory()
    
    return train_dataset, val_dataset

class ModelTrainer:
    def __init__(self, modality, healing_phase, base_dir):
        self.modality = modality
        self.healing_phase = healing_phase
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, modality, healing_phase)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 64
        self.batch_size = 8
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="fp16",
            log_with="tensorboard",
            project_dir=os.path.join(self.base_dir, 'logs')
        )
        
        # Initialize metrics tracking
        self.metrics_file = os.path.join(base_dir, 'training_metrics.csv')
        self.setup_metrics()
        self.best_val_loss = float('inf')
        self.patience = 3
        self.patience_counter = 0
        
        # Create metrics file with headers if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['modality', 'phase', 'epoch', 'train_loss', 'val_loss', 'fid', 'ssim'])
        
        # Load base model
        model_id = "runwayml/stable-diffusion-v1-5"
        self.load_models(model_id)
        self.setup_training()
        
        # Ensure all models are on the correct device
        self.to_device()
        
    def setup_data(self):
        """Setup data loaders with proper paths"""
        train_dataset, val_dataset = create_pinned_datasets(
            data_dir=self.data_dir,
            phase=self.healing_phase,
            modality=self.modality,
            tokenizer=self.tokenizer,
            cache_dir=os.path.join(self.base_dir, 'cache'),
            pin_memory=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        return train_loader, val_loader

    def setup_metrics(self):
        """Setup metrics with proper device placement"""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.metrics = {
                'fid': FrechetInceptionDistance(
                    feature=768,
                    normalize=True, # True for 0-1 range images
                ).to(self.device),
                'ssim': StructuralSimilarityIndexMeasure().to(self.device)
            }
            # Ensure metrics are in eval mode
            for metric in self.metrics.values():
                metric.eval()
                
        except ImportError:
            print("Warning: Some metrics unavailable. Install torchmetrics and pytorch-msssim.")
            self.metrics = None

    def to_device(self):
        """Ensure all models are on the correct device"""
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)
        # if hasattr(self, 'noise_scheduler'):
        #     self.noise_scheduler = self.noise_scheduler.to(self.device)

    def load_models(self, model_id):
        """Load all required models"""
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        # self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        
        # Load VAE with custom config
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        # Update VAE config if needed
        self.vae.config.sample_size = self.image_size
        
        # Load UNet with custom config
        # Configure UNet for smaller resolution
        unet_config = {
            "sample_size": 64,  # Set to target image size
            "in_channels": 4,
            "out_channels": 4,
            "down_block_types": (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            "block_out_channels": (320, 640, 1280, 1280),
            "layers_per_block": 2,
            "cross_attention_dim": 768,  # Match CLIP text encoder dimension
            "attention_head_dim": 8,
            "use_linear_projection": False,
            "only_cross_attention": False,
        }
        
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            **unet_config
        )
        
        # Freeze components we don't want to train
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        # Set UNet to training mode
        self.unet.train()
        if hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
    
    def setup_pipeline(self):
        """Setup pipeline with configuration"""
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32
        )
        
        # Update pipeline components
        pipeline.vae.config.sample_size = self.image_size
        pipeline.unet.config.sample_size = self.image_size
        
        return pipeline
    
    def setup_training(self):
        """Initialize optimizer and other training components"""
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )

    def split_dataset(self, full_dataset, train_ratio=0.8):
        """Split dataset into train and validation sets"""
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset

    def generate_validation_samples(self, encoder_hidden_states, num_samples=4):
        """Generate samples with optimized validation settings"""
        self.unet.eval()
        with torch.no_grad():
            # Start from random noise
            latents = torch.randn(
                (num_samples, self.unet.config.in_channels, self.image_size // 8, self.image_size // 8), # VAE reduces size by factor of 8
                device=self.device
            )
            
            # Prepare encoder hidden states once
            if encoder_hidden_states.shape[0] != num_samples:
                encoder_hidden_states = encoder_hidden_states.repeat(num_samples, 1, 1)
            encoder_hidden_states = encoder_hidden_states.to(self.device)
            
            # Use fewer timesteps for validation
            num_inference_steps = 20  # Much fewer than the default 1000
            self.noise_scheduler.set_timesteps(num_inference_steps)
            timesteps = self.noise_scheduler.timesteps.to(self.device)
            
            # Optional: Use larger steps for even faster inference
            timestep_spacing = len(timesteps) // 10  # Use only 10 steps
            timesteps = timesteps[::timestep_spacing]
            
            # Create timestep tensors once
            timestep_tensor = timesteps.unsqueeze(1).repeat(1, num_samples).reshape(-1)
            # Batch process all timesteps
            for t in timesteps:
                # Expand t for batch processing
                t_batch = t.repeat(num_samples)
                
                # Generate noise prediction
                noise_pred = self.unet(
                    latents,
                    t_batch,
                    encoder_hidden_states
                ).sample
                
                # Update latents
                latents = self.noise_scheduler.step(
                    noise_pred, 
                    t, 
                    latents
                ).prev_sample
            # Decode latents to images
            images = self.vae.decode(latents / self.vae.config.scaling_factor).sample

        self.unet.train()
        return images
    
    def validate(self, val_dataloader):
        """Run validation pass with proper device handling"""
        self.unet.eval()
        val_loss = 0
        num_batches = 0
        # Initialize lists for images
        real_images = []
        generated_images = []
        min_required_samples = 10
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                # Regular validation step
                latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 
                    self.noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=self.device
                )
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states
                ).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                val_loss += loss.item()
                num_batches += 1
                # Generate samples for metrics
                if len(real_images) < min_required_samples:
                    # Generate images
                    generated = self.generate_validation_samples(
                        encoder_hidden_states.detach(),  # Ensure detached tensor
                        num_samples=batch["pixel_values"].shape[0]
                    )
                    # Store real and generated images
                    real_images.extend(batch["pixel_values"].cpu())
                    generated_images.extend(generated.cpu())
                if len(real_images) >= min_required_samples:
                    break

        # Calculate average validation loss
        val_loss = val_loss / num_batches

        # Initialize metrics dictionary
        metrics_dict = {'val_loss': val_loss}
        # Calculate metrics if we have enough samples
        if self.metrics and len(real_images) >= min_required_samples:
            try:
                real_batch = torch.stack(real_images[:min_required_samples]).to(self.device)
                generated_batch = torch.stack(generated_images[:min_required_samples]).to(self.device)
                # Ensure proper image format for FID
                if real_batch.max() <= 1.0:
                    real_batch = (real_batch * 255).to(torch.uint8)
                    generated_batch = (generated_batch * 255).to(torch.uint8)
                # Calculate FID
                if 'fid' in self.metrics:
                    self.metrics['fid'].reset()
                    self.metrics['fid'].update(real_batch, real=True)
                    self.metrics['fid'].update(generated_batch, real=False)
                    metrics_dict['fid'] = self.metrics['fid'].compute().item()
                # Calculate SSIM
                if 'ssim' in self.metrics:
                    metrics_dict['ssim'] = self.metrics['ssim'](
                        generated_batch.float() / 255.0 if generated_batch.max() > 1.0 else generated_batch,
                        real_batch.float() / 255.0 if real_batch.max() > 1.0 else real_batch
                    ).item()
            except Exception as e:
                print(f"Error computing metrics: {str(e)}")
                metrics_dict.update({
                    'fid': float('nan'),
                    'ssim': float('nan')
                })
            
            # Clean up CUDA memory
            del real_batch, generated_batch
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

        self.unet.train()
        return metrics_dict
        
    def setup_training(self):
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
    def train(self, num_epochs=100, batch_size=1):
        
        self.batch_size = batch_size
        # Create full dataset first
        train_dataloader, val_dataloader = self.setup_data()
        
        # Prepare for distributed training
        self.unet, self.optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, train_dataloader, val_dataloader
        )
        
        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["pixel_values"].to(self.device)
                    ).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    # Add noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=self.device
                    )
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get text embeddings
                    encoder_hidden_states = self.text_encoder(
                        batch["input_ids"].to(self.device)
                    )[0]

                    # Predict noise
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states
                    ).sample

                    # Calculate loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    
                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item())
                    global_step += 1

            avg_train_loss = epoch_loss / len(train_dataloader)

            # Run validation every 3 epochs
            if (epoch + 1) % 3 == 0:
                metrics = self.validate(val_dataloader)
                metrics['train_loss'] = avg_train_loss
                
                # Log metrics
                self.log_metrics(epoch + 1, metrics)
                
                print(f"\nEpoch {epoch + 1} Metrics:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                
                # Save model if validation loss improved
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    self.patience_counter = 0
                    self.save_checkpoint(epoch + 1)
                else:
                    self.patience_counter += 1
                
                # Early stopping check
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    del train_dataloader, val_dataloader, self.unet, self.optimizer, self.accelerator
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                    break
                
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
            del noise, timesteps, noisy_latents, latents, encoder_hidden_states, noise_pred

    def log_metrics(self, epoch, metrics):
        """Log metrics to CSV file"""
        if self.accelerator.is_main_process:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.modality,
                    self.healing_phase,
                    epoch,
                    metrics.get('train_loss', 'N/A'),
                    metrics.get('val_loss', 'N/A'),
                    metrics.get('fid', 'N/A'),
                    metrics.get('ssim', 'N/A')
                ])

    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        if self.accelerator.is_main_process:
            pipeline = self.setup_pipeline()
            
            new_pipeline = StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.accelerator.unwrap_model(self.unet),
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=pipeline.feature_extractor,
                requires_safety_checker=False
            )
            
            save_path = os.path.join(
                self.base_dir, 
                'models',
                f'{self.modality}_{self.healing_phase}'
            )
            os.makedirs(save_path, exist_ok=True)
            new_pipeline.save_pretrained(save_path)
            
            print(f"\nSaved checkpoint to {save_path}")
            
            del pipeline, new_pipeline
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

def main():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    # Your original directory setup and main loop
    base_dirs = {
        "mac": "/Volumes/Expansion/DFUCalgary",
        "windows": "G:/DFUCalgary",
        "cc": "/project/6086937/basirire/multimodal",
        "onedrive": "C:/Users/90rez/OneDrive - University of Toronto/PhDUofT/ZivotData"
    }
    
    base_dir = next((dir_path for dir_path in base_dirs.values() if os.path.exists(dir_path)), None)
    if base_dir is None:
        raise RuntimeError("No valid directory found!")
    
    base_dir = os.path.join(base_dir, 'Codes/MultimodalClassification/ImageGeneration')
    
    modalities = ['rgb', 'depth_map', 'thermal_map']
    healing_phases = ['I', 'P', 'R']
    
    for modality in modalities:
        for phase in healing_phases:
            if modality == 'rgb' and (phase == 'I' or phase == 'P' or phase == 'R'):
                continue
            if modality == 'depth_map' and (phase == 'I' or phase == 'P' or phase == 'R'):
                continue
            if modality == 'thermal_map' and (phase == 'I'):
                continue
            while True:
                try:
                    if 'pipeline' in locals(): print('remaining pipeline deleted'); del pipeline
                    if 'new_pipeline' in locals(): print('remaining new_pipeline deleted'); del new_pipeline
                    if 'model_to_save' in locals(): del model_to_save
                    if 'unwrapped_model' in locals(): del unwrapped_model
                    if 'trainer' in locals(): del trainer
                    if 'dataset' in locals(): del dataset
                    if 'dataloader' in locals(): del dataloader
                    if 'model' in locals(): print('remaining model deleted'); del model
                    if 'optimizer' in locals(): del optimizer
                    if 'scheduler' in locals(): del scheduler
                    if 'text_encoder' in locals(): del text_encoder
                    if 'vae' in locals(): del vae
                    if 'unet' in locals(): del unet
                    if 'tokenizer' in locals(): del tokenizer
                    if 'prompt_embeds' in locals(): del prompt_embeds
                    if 'text_inputs' in locals(): del text_inputs
                    print(f"\nTraining model for {modality} - {phase}")
                    trainer = ModelTrainer(modality, phase, base_dir)
                    trainer.train(num_epochs=50, batch_size=8)
                    del trainer
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                    break # Exit while loop
                except Exception as e:
                    print(f"Error occurred: {str(e)}")
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()