import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm
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
    
class WoundImageGenerator:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {model_path}...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Set image size in pipeline
        self.pipeline.vae.config.sample_size = 64
        self.pipeline.unet.config.sample_size = 64
        
        # Setup prompt generator
        self.prompt_generator = PromptGenerator()
    
    def generate_images(self, modality, phase, num_images, output_size=(64, 64), 
                       output_dir="generated_images", seed=42):
        """
        Generate images for specified modality and phase
        Args:
            modality (str): 'rgb', 'depth_map', or 'thermal_map'
            phase (str): 'I', 'P', or 'R'
            num_images (int): Number of images to generate
            output_size (tuple): Size of output images (height, width)
            output_dir (str): Directory to save generated images
            seed (int): Random seed for reproducibility
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get prompt for this modality and phase
        prompt, negative_prompt = self.prompt_generator.generate_prompt(modality, phase)
        
        # Set random seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"\nGenerating {num_images} images for {modality} - {phase}")
        print(f"Using prompt: {prompt}")
        
        # Generate images
        for i in tqdm(range(num_images)):
            # Generate image
            with torch.no_grad():
                image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=200,
                    generator=generator,
                    height=64,  # Initial generation at model's training size
                    width=64
                ).images[0]
            
            # Resize if needed
            if output_size != (64, 64):
                image = image.resize(output_size, Image.Resampling.LANCZOS)
            
            # Save image
            filename = f"{modality}_{phase}_generated_{i+1:03d}.png"
            image.save(os.path.join(output_dir, filename))

# Example usage
if __name__ == "__main__":
    # Set paths
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
    
    # Setup generator for specific modality and phase
    modality = "rgb"
    for phase in ["I", "P", "R"]:
        model_path = os.path.join(base_dir, "models_5_7", f"{modality}_{phase}")
        
        # Create generator
        generator = WoundImageGenerator(model_path)
        
        # Generate images
        generator.generate_images(
            modality=modality,
            phase=phase,
            num_images=50,  # Generate 20 images
            output_size=(128, 128),  # Upscale to 96x96
            output_dir=os.path.join(base_dir, "generated_images_5_7_v2", f"{modality}_{phase}"),
            seed=42  # Set seed for reproducibility
        )