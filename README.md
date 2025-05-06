# PromptTo3D

PromptTo3D is a Python-based prototype that generates 3D models from textual prompts or images using OpenAI's Shape-E framework. This project leverages diffusion models to create high-quality 3D representations, which can be rendered as images or exported as 3D mesh files in `.stl` and `.obj` formats.

## Features

- **Text-to-3D Model Generation**: Generate 3D models from descriptive text prompts.
- **Image-to-3D Model Generation**: Generate 3D models from input images.
- **Customizable Rendering**: Supports different rendering modes (`nerf` or `stf`) and adjustable output sizes.
- **Export Options**: Save generated 3D models as `.stl` or `.obj` files.

## Samples GIFs

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
            <td align="center">
                <img src="Sample GIFs/mountain.gif" alt="A mountain">
            </td>
            <td align="center">
                <img src="Sample GIFs/tree.gif" alt="A tree">
            </td>
            <td align="center">
                <img src="Sample GIFs/chair.gif" alt="A chair">
            </td>
            <td align="center">
                <img src="Sample GIFs/cow.gif" alt="A cow">
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

## Installation and Usage

### Note: 
As this project was developed and extensively tested in a Google Colab notebook, it is recommended that you deploy and test it in a colab notebook as well.
From now on, the installation and usage will be tailored for a google colab notebook

1. Clone the Shape-E repository:
   ```bash
   !git clone https://github.com/harsh16629/PromptTo3D.git
   ```

2. Run the Main.ipynb file and clone the shap-e repo 
```bash
!git clone https://github.com/openai/shap-e 
```

3. Install the required dependencies using:
```bash
%cd shap-e             
!pip install -e .
```

4. Import required scripts:
```python
import txtTo3D, imgTo3D
```
5. For Text-to-3D Model Generation
```python
# For text to 3D generation
prompts = []
while True:
  prompt = input("Enter a prompt for 3D generation (or type 'done' to finish): ")
  if prompt.lower() == 'done':
    break
  prompts.append(prompt)

for prompt in prompts:
  txtTo3D.txtTo3d(prompt, size_=128, render_mode_='nerf')
```
6.  For Image-to-3D Model Generation
```python
# For image to 3D generation
# To get the best result, remove the background from the image, or use a plain background.
from google.colab import files
import io
from PIL import Image

uploaded = files.upload()

# Process each uploaded file
for filename, byte_content in uploaded.items():
  # Convert the byte content to a PIL Image
  image = Image.open(io.BytesIO(byte_content))

  # Call the function with your image
  imgTo3D.imgTo3D(image, size_=264, render_mode_='nerf')
```
7. Rendering Modes
- `nerf`: Neural Radiance Fields rendering (default).
- `stf`: Surface rendering.

## Output Files
The generated 3D models are saved in the current working directory as:

- `text_generated_mesh_<index>.stl`
- `text_generated_mesh_<index>.obj`
- `image_generated_mesh_<index>.stl`
- `image_generated_mesh_<index>.obj`

## Sample Input and Output Screenshots

<table>
  <tr>
    <td><img src="IO samples\image_input_and_output.png" width="800" height="250"></td>
  </tr>
  <tr>
    <td><img src="IO samples\text_input_and_output.png" width="800" height="250"></td>
  </tr>
</table>

## File Structure
```bash
.
├── IO samples/
    ├── image_input_and_output.png
    └── text_input_and_output.png
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
