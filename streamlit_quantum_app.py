import importlib.util
import sys

# Import from the Image_Generation.py file
spec = importlib.util.spec_from_file_location("Image_Generation", "Image_Generation.py")
image_generation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_generation_module)

# Re-export everything from the loaded module
for attr_name in dir(image_generation_module):
    if not attr_name.startswith('_'):
        globals()[attr_name] = getattr(image_generation_module, attr_name)

# Legacy alias for downstream imports