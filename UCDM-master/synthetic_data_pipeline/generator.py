import os
import sys
import time
import math
import torch
import random
import PIL
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms, models, utils
from torchvision.transforms import ToTensor, Normalize
from pathlib import Path
from typing import Optional, Tuple, Union
import torch.nn as nn
from torch import autocast, inference_mode
import torch.utils.checkpoint
import torch.utils.data as data
from diffusers import StableDiffusionPipeline
from ddim import DDIMScheduler
import matplotlib.pyplot as plt


class TrainingConfig:
    """
    Configuration class for training parameters used in Stable Diffusion models.

    Attributes:
        height (int): Height of the input image (default: 512).
        width (int): Width of the input image (default: 512).
        num_inference_steps (int): Number of denoising steps for generation (default: 20).
        guidance_scale (float): Classifier-free guidance scale (default: 7.5).
        generator (torch.Generator): Seed generator for initial latent noise (default: torch.manual_seed(0)).
        batch_size (int): Number of samples in each batch (default: 1).
    """

    def __init__(self):
        self.height = 512  # Default height for Stable Diffusion
        self.width = 512  # Default width for Stable Diffusion
        self.num_inference_steps = 20  # Number of denoising steps
        self.guidance_scale = 7.5  # Classifier-free guidance scale
        self.generator = torch.manual_seed(0)  # Seed generator for initial latent noise
        self.batch_size = 1  # Batch size


    def __repr__(self):
        return f"TrainingConfig(height={self.height}, width={self.width}, num_inference_steps={self.num_inference_steps}, " \
               f"guidance_scale={self.guidance_scale}, batch_size={self.batch_size})"


class LoadGenerator(nn.Module):
    """
    This class defines the generator for the model that leverages Stable Diffusion for generating images.
    It includes methods for loading models, processing images, and generating outputs with specified labels.
    """

    def __init__(self, known_class_names, train_device, batch_num=5,model_id="/data/stable-diffusion-2-base", image_size=(32, 32)):
        """
        Initialize the generator with the given configuration.

        Args:
            known_class_names (list): List of known class names for generating text embeddings.
            train_device (torch.device): The device to use for training (e.g., "cuda" or "cpu").
            model_id (str): Path or identifier for the pre-trained model.
            image_size (tuple): Target image size for preprocessing.
        """
        super(LoadGenerator, self).__init__()
        
        self.model_id = model_id
        self.config = TrainingConfig()  
        self.device = train_device
        self.pipe = self.load_model()
        self.uncond_embeddings = self.load_unconditional_prompt()
        self.known_class_names = known_class_names
        self.save_fig = True
        self.save_batch_num = 0
        self.batch_num = batch_num
        self.transform_layers = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.inv_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage()
        ])

    def load_model(self):
        """Load the Stable Diffusion model and configure it."""
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=DDIMScheduler(
                beta_end=0.012,
                beta_schedule="scaled_linear",
                beta_start=0.00085
            ),
            torch_dtype=torch.float16,
            revision="fp16"
        )
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False 
        for param in pipe.unet.parameters():
            param.requires_grad = False  
        for param in pipe.vae.parameters():
            param.requires_grad = False  

        pipe.vae = pipe.vae.to(torch.float16)
        pipe = pipe.to(self.device)
        pipe.scheduler.set_timesteps(self.config.num_inference_steps)
        
        print(f"Generator loaded, num_inference_steps: {pipe.scheduler.num_inference_steps}")
        return pipe

    def load_unconditional_prompt(self):
        """Load the unconditional embeddings."""
        uncond_input = self.create_token("", padding=True)
        uncond_embeddings = self.pipe.text_encoder(uncond_input.to(self.device))[0]
        return uncond_embeddings.unsqueeze(0)

    def create_token(self, prompt, padding=False):
        """Create token IDs for the given prompt."""
        prompt_input_ids = self.pipe.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length
        ).input_ids
    
        if not padding:
            return prompt_input_ids     
        else:
            input_ids = self.pipe.tokenizer.pad(
                {"input_ids": prompt_input_ids},
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids
            return input_ids.view(1, input_ids.shape[0])

    def vae_encoder(self, images):
        """Encode images using the VAE encoder to obtain latent representations."""
        init_latents = self.pipe.vae.encode(images.to(self.device, dtype=torch.float16)).latent_dist.sample(self.config.generator)
        init_latents = 0.18215 * init_latents
        return init_latents

    def inference_step(self, latents_, text_embedding, peta=1, neta=0.2):
        """
        Perform a single inference step for both positive and negative  latents.

        Args:
            latents_ (torch.Tensor): Latent vectors for the image.
            text_embedding (torch.Tensor): Text embeddings for conditioning.
            eta (float): Hyperparameter for controlling noise. 
        Returns:
            tuple: Lists of unconditioned and conditioned latents.
        """
        uncondition_latents_list = []
        condition_latents_list = []
        negative_latents = latents_.clone()

        with autocast("cuda"), inference_mode():
            timesteps = self.pipe.scheduler.timesteps.flip(0)
            for t in timesteps:
                noise_pred = self.pipe.unet(negative_latents, t, encoder_hidden_states=text_embedding[:, 1, :, :]).sample
                negative_latents = self.pipe.scheduler.con_inverse_step(noise_pred, t, negative_latents).next_sample

        noise = torch.randn(latents_.shape, generator=self.config.generator, dtype=latents_.dtype).to(self.device)
        positive_latents = self.pipe.scheduler.add_noise(latents_, noise, torch.LongTensor([self.pipe.scheduler.timesteps[0]]))

        with autocast("cuda"), inference_mode():
            for t in self.pipe.scheduler.timesteps:
                with torch.no_grad():
                    negative_latents = self.pipe.scheduler.scale_model_input(negative_latents, t)
                    positive_latents = self.pipe.scheduler.scale_model_input(positive_latents, t)

                    negative_noise_pred = self.pipe.unet(negative_latents, t, encoder_hidden_states=text_embedding[:, 0, :, :]).sample
                    positive_uncon_noise_pred = self.pipe.unet(positive_latents, t, encoder_hidden_states=text_embedding[:, 0, :, :]).sample
                    positive_con_noise_pred = self.pipe.unet(positive_latents, t, encoder_hidden_states=text_embedding[:, 1, :, :]).sample
                    positive_noise_pred = positive_uncon_noise_pred + self.config.guidance_scale * (positive_con_noise_pred - positive_uncon_noise_pred)

                    negative_latents = self.pipe.scheduler.step(negative_noise_pred, t, negative_latents, generator=self.config.generator, eta=neta).prev_sample
                    positive_latents = self.pipe.scheduler.step(positive_noise_pred, t, positive_latents, generator=self.config.generator, eta=peta).prev_sample

                    negative_latents = negative_latents.detach()
                    positive_latents = positive_latents.detach()

                    if t == self.pipe.scheduler.timesteps[-1]:
                        condition_latents_list.append(positive_latents)
                        uncondition_latents_list.append(negative_latents)

        return uncondition_latents_list, condition_latents_list

    def show_latents(self, latents_list, size=512):
        """Convert latents to images and return a list of PIL images."""
        images_save = []
        with torch.no_grad():
            for latents in latents_list:
                images = self.pipe.decode_latents(latents.to(torch.float16))
                im = self.pipe.numpy_to_pil(images)
                for image in im:
                    images_save.append(image.resize((512, 512)))
        return images_save

    def visualize_images(self, generated_images):
        """Visualize generated images and save them to disk."""
        path = "./outputs"
        if not os.path.exists(path):
            os.makedirs(path)

        k, _ = divmod(len(generated_images), 3)
        original = generated_images[:k]
        unconditioned = generated_images[k:k * 2]
        conditioned = generated_images[2 * k:]

        for i, (ori, uncon, con) in enumerate(zip(original, unconditioned, conditioned)):
            fig, axes = plt.subplots(1, 3, figsize=(20, 10))
            axes[0].imshow(ori)
            axes[1].imshow(uncon)
            axes[2].imshow(con)

            save_path = os.path.join(path, f"image_{self.save_batch_num}_{i}.png")
            fig.savefig(save_path)
            plt.show()

    def save_images(self, outputs):
        """Save the generated images."""
        self.visualize_images(self.show_latents(outputs))

    def label_to_embedding(self, labels):
        """Generate text embeddings from class labels."""
        targets = [self.create_token(f"A photo of a {self.known_class_names[i]}", padding=True) for i in labels]
        targets = torch.cat(targets, dim=0)
        cond_embeddings = self.pipe.text_encoder(targets.to(self.device))[0]
        cond_embeddings = cond_embeddings.unsqueeze(1)
        text_embeddings = torch.cat([self.uncond_embeddings.repeat(cond_embeddings.shape[0], 1, 1, 1), cond_embeddings], dim=1)
        return text_embeddings

    def decoder_transform(self, latents_list):
        """Transform latents back to images after decoding."""
        images_list = []
        with torch.no_grad():
            for latents in latents_list:
                images = self.pipe.decode_latents(latents.to(torch.float16))
                images = self.pipe.numpy_to_pil(images)
                for image in images:
                    images_list.append(self.transform_layers(image).unsqueeze(0))
        return images_list

    def tensor_to_images(self, tensor_batch):
        """Convert a batch of tensors to images."""
        images = []
        for tensor in tensor_batch:
            tensor = tensor.cpu().clone()
            image = self.inv_transform(tensor)
            images.append(image)
        return images

    def preprocess_image(self, img):
        """Preprocess input image for model compatibility."""
        image = img.convert("RGB").resize((512, 512))
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # Ensure dimensions are multiples of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float16) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        img = 2.0 * image - 1.0  # Normalize to [-1, 1]
        return img.squeeze(0)

    def process_batch_images(self, tensor_batch):
        """Process a batch of images for model input."""
        images = self.tensor_to_images(tensor_batch)
        processed_tensors = [self.preprocess_image(img) for img in images]
        processed_batch = torch.stack(processed_tensors)
        return processed_batch

    def generate(self, images_, labels=None):
        """
        Generate images based on input images and class labels.

        Args:
            images_ (list): List of images to process.
            batch_num (int): Batch size for processing. Defaults to 30.
            labels (list): List of labels for conditioning the generation.

        Returns:
            tuple: Processed unconditioned and conditioned tensors, and labels.
        """
        images = self.process_batch_images(images_)
        latents_ = self.vae_encoder(images)
        labels_ = torch.arange(len(self.known_class_names))
        flabels = labels_.repeat(images.shape[0])
        text_embeddings = self.label_to_embedding(flabels)
        latents = latents_.repeat(len(self.known_class_names), 1, 1, 1)
        chunk_num = math.ceil(latents.shape[0] / self.batch_num)
        negative_class_tensor = []
        positive_class_tensor = []
        
        for i, (latent, text) in enumerate(zip(latents.chunk(chunk_num), text_embeddings.chunk(chunk_num))):
            negative_noise_features, positive_noise_features = self.inference_step(latent, text)
            negative_class_tensor.extend(self.decoder_transform(negative_noise_features))
            positive_class_tensor.extend(self.decoder_transform(positive_noise_features))

            if self.save_fig:
                latents_list = list(latent.chunk(latent.shape[0]))
                outputs = latents_list + negative_noise_features + positive_noise_features
                self.save_images(outputs)
                self.save_batch_num += 1
                if self.save_batch_num > 2:
                    self.save_fig = False

        negative_class_tensor_ = torch.cat(negative_class_tensor, dim=0)
        positive_class_tensor_ = torch.cat(positive_class_tensor, dim=0)

        return negative_class_tensor_, positive_class_tensor_, flabels

