import os
from PIL import Image

# Create the necessary directories
output_dir = '../static/placeholder/400'
os.makedirs(output_dir, exist_ok=True)

# Create a placeholder image (400x320) with a solid color
placeholder_image = Image.new('RGB', (400, 320), color=(200, 200, 200))

# Save the placeholder image to the specified path
placeholder_image.save(os.path.join(output_dir, '320.png'))