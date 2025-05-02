# Import libraries
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # Use a dedicated GPU for better performance


xm = load_model('transmitter', device=device)    # Transimitter model; maps latent encodings into 3D space
model = load_model('text300M', device=device)    # Text encoding model; converts textual prompt to vector
diffusion = diffusion_from_config(load_config('diffusion'))  # Core generative model


from shap_e.util.notebooks import decode_latent_mesh

# Main function to generate 3D models
def txtTo3d(prompt_, size_=64, render_mode_ ='nerf'):
    batch_size = 1
    guidance_scale = 15.0                                  # Higher value results more precise output
    prompt = prompt_

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),    # Setting the conditioning prompt for all items in the batch
        progress=True,                                     # Shows a progress bar
        clip_denoised=True,                                # Denoising for more coherent output
        use_fp16=True,                                     # Uses 16-bit floating point instead of 32-bit for faster compute time
        use_karras=True,                                   # Keras sampling for better denoising
        karras_steps=64,                                   # Number of diffusion steps
        sigma_min=1e-3,                                    # Starting noise level
        sigma_max=160,                                     # Final noise level
        s_churn=0                                          # No additional stochastic noise added during sampling
    )
    render_mode = render_mode_                             # Rendering mode for the output images; 'nerf' or 'stf'
    size = size_                                           # Size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)

    # Loop to display 3D gif
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        display(gif_widget(images))


    # Loop to save the generated mesh
    for i, latent in enumerate(latents):
      t = decode_latent_mesh(xm, latent).tri_mesh()
      with open(f'text_generated_mesh_{i}.stl', 'wb') as f:
          t.write_ply(f)
      with open(f'text_generated_mesh_{i}.obj', 'w') as f:
          t.write_obj(f)

