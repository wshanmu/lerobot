"""
1. Load community `LeRobotDataset` from Huggingface.
2. Get a single image from an episode
3. Using Meta SAM model to segment the image
4. Assign Synthetic `Temperatures` to Each Class
    a. Core-to-Edge Gradient
    b. Texture-Based Heat Patterns: Introduce slight noise or spot variations so the heat distribution isn't perfectly uniform
5. Apply a Thermal Color Palette
"""