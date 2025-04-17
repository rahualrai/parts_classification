import rembg
import numpy as np
from PIL import Image

# Load the input image
input_image = Image.open('data/parts/train/connector/image_03.jpg')


input_array = np.array(input_image)

# Apply background removal using rembg
output_array = rembg.remove(input_array)

# Create a PIL Image from the output array
output_image = Image.fromarray(output_array)

# Save the output image
output_image.save('data/parts/test/gear/image_01_bg.png')
