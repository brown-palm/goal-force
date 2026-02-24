"""
MIT License

Copyright (c) 2025 Nate Gillman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from flask import Flask, render_template, request, send_file, jsonify
import os
import io
import base64
import cv2
import json
import csv
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from system_prompts import SYS_PROMPT_I2V

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Create necessary directories
output_dir = "datasets/custom_test"
os.makedirs(output_dir, exist_ok=True)

from mimetypes import guess_type
def image_to_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

@app.route('/')
def index():
    return render_template('index_goal_force.html')

@app.route('/crop', methods=['POST'])
def crop_image():
    file = request.files['image']
    x = float(request.form['x'])
    y = float(request.form['y'])
    width = float(request.form['width'])
    height = float(request.form['height'])
    aspect_ratio = float(request.form['aspect_ratio'])
    
    # Get original filename for later use
    original_filename = file.filename
    
    # Open and crop the image
    img = Image.open(file.stream)
    cropped = img.crop((x, y, x + width, y + height))
    
    # Resize based on aspect ratio
    if abs(aspect_ratio - 1.7333) < 0.01:
        target_size = (832, 480)  # Width x Height for 1.5 aspect ratio
    # elif abs(aspect_ratio - 1.7708) < 0.01:
    #     target_size = (1360, 768)  # Width x Height for 1.7708 aspect ratio
    else:
        # Default - don't resize
        target_size = None
    
    # Apply resizing if target size is specified
    if target_size:
        resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    else:
        resized = cropped

    # Convert to RGB to remove the alpha channel before saving
    # This will fix the 'rgba' output issue
    resized = resized.convert('RGB')
    
    # Generate a new filename with underscore prefix
    base_name = os.path.basename(original_filename)
    new_filename = f"_{os.path.splitext(base_name)[0]}.png"
    file_dir = os.path.join(output_dir, "images")
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, new_filename)
    
    # Save the image to disk
    resized.save(file_path)
    
    # Save to bytes for direct response
    img_io = io.BytesIO()
    resized.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Return both the file and its path
    response = send_file(img_io, mimetype='image/png')
    response.headers['X-Filename'] = new_filename
    response.headers['X-Filepath'] = file_path
    return response

@app.route('/optimize_prompt', methods=['POST'])
def optimize_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    image_path = data.get('image_path', '')
    
    if not os.path.exists(image_path):
        return jsonify({'error': f'Image not found: {image_path}'}), 404
    
    # Use the image-to-video (i2v) prompt optimization logic
    try:
        client = OpenAI(api_key=api_key)
         
        # Call the API with image and prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"{SYS_PROMPT_I2V}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_url(image_path),
                            },
                        },
                    ],
                },
            ],
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        
        optimized_prompt = response.choices[0].message.content
        return jsonify({'optimized_prompt': optimized_prompt})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_pixel', methods=['POST'])
def get_pixel():
    data = request.json
    image_path = data.get('image_path')
    x = data.get('x')
    y = data.get('y')
    displayed_width = data.get('displayed_width')
    displayed_height = data.get('displayed_height')
    
    # Open the image with OpenCV
    img = cv2.imread(image_path)
    actual_height, actual_width = img.shape[:2]
    
    # Calculate the scale between displayed and actual image
    scale_x = actual_width / displayed_width
    scale_y = actual_height / displayed_height
    
    # Convert click coordinates to actual image coordinates
    # The y coordinate needs to be measured from the top of the displayed image
    # and then convert to the bottom-left origin system
    actual_x = int(x * scale_x)
    
    # First calculate the y-coordinate in the top-left origin system
    actual_y_top = int(y * scale_y)
    
    # Then convert to bottom-left origin
    # This is what was going wrong - we need to handle the conversion correctly
    y_bottom_left = actual_height - actual_y_top
    
    # Make sure coordinates are in bounds
    actual_x = max(0, min(actual_x, actual_width - 1))
    y_bottom_left = max(0, min(y_bottom_left, actual_height - 1))
    
    return jsonify({
        'x': actual_x,
        'y': y_bottom_left,
        'width': actual_width,
        'height': actual_height
    })

@app.route('/write_csv', methods=['POST'])
def write_csv():
    data = request.json
    image_path = data.get('image_path')
    caption = data.get('caption')
    projectile_coordx = data.get('projectile_coordx')
    projectile_coordy = data.get('projectile_coordy')
    target_coordx = data.get('target_coordx') 
    target_coordy = data.get('target_coordy') 
    projectile_force_angle = data.get('projectile_force_angle', 0.0)
    projectile_force_magnitude = data.get('projectile_force_magnitude', 350.0)
    target_indirect_force_angle = data.get('target_indirect_force_angle', 0.0) # ADDED
    target_indirect_force_magnitude = data.get('target_indirect_force_magnitude', 350.0) # ADDED
    prev_coord_x = data.get('prev_coord_x')
    prev_coord_y = data.get('prev_coord_y')
    prev_obj_counter = data.get('prev_obj_counter', 0)
    prev_prompt_counter = data.get('prev_prompt_counter', 0)
    
    # Get image dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Determine the object and prompt counters
    obj_counter = prev_obj_counter
    prompt_counter = prev_prompt_counter
    
    # If coordinates have changed, increment object counter and reset prompt counter
    # Note: This logic still bases the "object" on the projectile coordinate
    if prev_coord_x is not None and prev_coord_y is not None:
        if int(projectile_coordx) != int(prev_coord_x) or int(projectile_coordy) != int(prev_coord_y):
            obj_counter += 1
            prompt_counter = 1
        else:
            # Coordinates are the same, increment prompt counter
            prompt_counter += 1
    else:
        # First time saving, start with obj1_prompt1
        obj_counter = 1
        prompt_counter = 1
    
    # Generate CSV name based on image name
    image_basename = os.path.basename(image_path)
    image_name = os.path.splitext(image_basename)[0]
    csv_filename = f"{image_name}_obj{obj_counter}_prompt{prompt_counter}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # --- START: MODIFIED CSV WRITING ---
    
    # Create the two rows
    row1 = {
        'image': image_basename,
        'projectile_force_angle': float(projectile_force_angle),
        'projectile_force_magnitude': float(projectile_force_magnitude),
        'projectile_coordx': int(projectile_coordx),
        'projectile_coordy': int(projectile_coordy),
        'projectile_mass': -1, 
        'target_indirect_force_angle': -1.0, 
        'target_indirect_force_magnitude': -1.0, 
        'target_coordx': int(target_coordx), 
        'target_coordy': int(target_coordy), 
        'target_mass': -1, 
        'width': width,
        'height': height,
        'caption': caption,
    }
    
    row2 = {
        'image': image_basename,
        'projectile_force_angle': -1.0,
        'projectile_force_magnitude': -1.0,
        'projectile_coordx': int(projectile_coordx),
        'projectile_coordy': int(projectile_coordy),
        'projectile_mass': -1, 
        'target_indirect_force_angle': float(target_indirect_force_angle), # UPDATED
        'target_indirect_force_magnitude': float(target_indirect_force_magnitude), # UPDATED
        'target_coordx': int(target_coordx), 
        'target_coordy': int(target_coordy), 
        'target_mass': -1, 
        'width': width,
        'height': height,
        'caption': caption,
    }
    
    # Write the CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'image', 'projectile_force_angle', 'projectile_force_magnitude', 
            'projectile_coordx', 'projectile_coordy', 'projectile_mass', 
            'target_indirect_force_angle', 'target_indirect_force_magnitude', 
            'target_coordx', 'target_coordy', 'target_mass', 
            'width', 'height', 'caption'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        # writer.writerow(row1) # row1: direct force (e.g. Force Prompting)
        writer.writerow(row2) # row2: indirect force (e.g. Goal Force)
    
    # --- END: MODIFIED CSV WRITING ---
    
    return jsonify({
        'success': True,
        'csv_path': csv_path,
        'obj_counter': obj_counter,
        'prompt_counter': prompt_counter
    })

if __name__ == '__main__':
    app.run(debug=True)