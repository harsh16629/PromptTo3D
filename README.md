# PromptTo3D

PromptTo3D is a Python-based prototype that generates 3D models from textual prompts or images using OpenAI's Shape-E framework. This project leverages diffusion models to create high-quality 3D representations, which can be rendered as images or exported as 3D mesh files in `.stl` and `.obj` formats.

## Features

- **Text-to-3D Model Generation**: Generate 3D models from descriptive text prompts.
- **Image-to-3D Model Generation**: Generate 3D models from input images.
- **Customizable Rendering**: Supports different rendering modes (`nerf` or `stf`) and adjustable output sizes.
- **Export Options**: Save generated 3D models as `.stl` or `.obj` files.

## Samples

<table>
    <body>
        <tr>
            <td align="center">
                <img src="Sample GIFs/red_apple.gif" alt="A red apple">
            </td>
            <td align="center">
                <img src="Sample GIFs/yellow_banana.gif" alt="A yellow banana">
            </td align="center">
            <td align="center">
                <img src="Sample GIFs/building.gif" alt="A building">
            </td>
          <td align="center">
                <img src="Sample GIFs/cat.gif" alt="A cat">
            </td>
        </tr>
    </body>
</table>

## Prerequisites

- Python 3.8 or higher
- A CUDA-compatible GPU (recommended for faster performance)
- Required Python libraries:
  - `torch`
  - `shap-e`

## Installation

1. Clone the Shape-E repository:
   ```bash
   git clone https://github.com/openai/shap-e
   cd shap-e
   ```
2. Install the required dependencies:
   ```bash
   pip install -e .
   ```
3. Navigate back to the project directory:
   ```bash
   cd ..
   ```

## Usage
1. Text-to-3D Model Generation
To generate a 3D model from a text prompt, use the txtTo3D module:
```python
import txtTo3D

# Example: Generate a 3D model of a red apple with a green leaf
txtTo3D.txtTo3d('A red apple with a green leaf')
```
2.  Image-to-3D Model Generation
To generate a 3D model from an image, use the imgTo3D module:
```python
import imgTo3D

# Load your image
image = imgTo3D.load_image("PATH_TO_YOUR_IMAGE")

# Generate a 3D model
imgTo3D.imgTo3D(image, size_=64, render_mode_='nerf')
```
3. Rendering Modes
- `nerf`: Neural Radiance Fields rendering (default).
- `stf`: Surface rendering.

## Output Files
The generated 3D models are saved in the current working directory as:

- `text_generated_mesh_<index>.stl`
- `text_generated_mesh_<index>.obj`
- `image_generated_mesh_<index>.stl`
- `image_generated_mesh_<index>.obj`

## File Structure
```bash
.
├── Sample GIFs/
    ├── red_apple.gif
    ├── yellow_banana.gif
    ├── building.gif
    └── cat.gif
├── LICENSE.txt          # MIT License
├── Main.ipynb           # Jupyter Notebook for running the prototype
├── README.md            # Project documentation
├── txtTo3D.py           # Script for text-to-3D model generation
└── imgTo3D.py           # Script for image-to-3D model generation
```
## Dependencies
The project uses the following Shape-E modules:

- `shap_e.diffusion.sample`: For sampling latents.
- `shap_e.diffusion.gaussian_diffusion`: For diffusion model configuration.
- `shap_e.models.download`: For loading pre-trained models and configurations.
- `shap_e.util.notebooks`: For rendering and displaying 3D outputs.
- `shap_e.util.image_util`: For loading input images.

## License
This project is licensed under the MIT License. See LICENSE.txt for details.

## Acknowledgments
This project is built on OpenAI's Shape-E framework. Special thanks to OpenAI for providing the tools and models used in this prototype.

## Notes
For best results with image-to-3D generation, use images with plain or transparent backgrounds.
Ensure that your GPU drivers and CUDA toolkit are up-to-date for optimal performance.
