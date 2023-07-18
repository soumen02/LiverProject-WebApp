import shutil
import subprocess
from flask import Flask, redirect, request, jsonify, send_file, render_template, url_for, session
import numpy as np
import requests
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import nibabel as nib


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # you can put any string here, but make sure it's hard to guess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your current directory
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')

# Make sure that the directory exists; if not, create it
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    # session.clear()
    return render_template("home.html", pv_filename=session.get('pv_filename', None),
                           ap_filename=session.get('ap_filename', None),
                           vp_filename=session.get('vp_filename', None))
def create_bounding_box(input_mask_path, output_mask_path):
    # Load the mask
    mask = nib.load(input_mask_path)
    mask_data = mask.get_fdata()

    # Create the bounding box mask
    min_x, max_x, min_y, max_y, min_z, max_z = compute_bounding_box(mask_data)

    # Create new mask filled with zeros
    new_mask_data = np.zeros(mask_data.shape)

    # Fill the box with ones
    new_mask_data[min_x:max_x, min_y:max_y, min_z:max_z] = 1

    # Save the mask
    new_mask = nib.Nifti1Image(new_mask_data, mask.affine, mask.header)
    nib.save(new_mask, output_mask_path)

def compute_bounding_box(mask):
    idx = np.where(mask != 0)
    return min(idx[0]), max(idx[0]), min(idx[1]), max(idx[1]), min(idx[2]), max(idx[2])

def crop_with_fslmaths(input_volume_path, bounding_box_path, output_volume_path):
    # Use fslmaths to crop the input volume based on the bounding box mask
    subprocess.run(['fslmaths', input_volume_path, '-mas', bounding_box_path, output_volume_path], check=True)

def process_volume(volume_path, output_dir):
    # Apply TotalSegmentator
    output_folder_name = os.path.join(output_dir, 'TotalSegmentator_output')
    subprocess.run(['TotalSegmentator', '-i', volume_path, '-o', output_folder_name, '--roi_subset', 'liver'])

    print(f'TotalSegmentator Applied {volume_path}...')

    # Parse into the folder, extract the liver label
    liver_mask_path = os.path.join(output_folder_name, 'liver.nii.gz')

    # Create a box the shape of the liver
    box_mask_path = os.path.join(output_dir, 'box_mask.nii.gz')
    create_bounding_box(liver_mask_path, box_mask_path)

    print(f'Bounding Box Created {volume_path}...')

    # Crop the volume using fslmaths
    cropped_volume_path = os.path.join(output_dir, 'cropped_' + os.path.basename(volume_path))
    crop_with_fslmaths(volume_path, box_mask_path, cropped_volume_path)

    print(f'Volume Cropped {volume_path}...')

    return cropped_volume_path

def dcm_to_nii(dcm_path, dcm_filename):
    nii_filename = os.path.splitext(dcm_filename)[0]
    subprocess.run(['dcm2niix', '-z', 'y', '-f', nii_filename, dcm_path], check=True)

    nii_filename = os.path.splitext(nii_filename)[0] + '.nii.gz'
    nii_path = os.path.join(os.path.dirname(dcm_path), nii_filename)
    
    return nii_path, nii_filename

def infer_model(model, session_key):
    # Check if a file was posted
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    # If the user does not select a file, the browser might
    # submit an empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    upload_folder  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_folder)
    os.makedirs(upload_dir, exist_ok=True)

    # Save the uploaded file to the upload folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    
    # if volume is .dcm file, convert to .nii.gz
    if filename.endswith('.dcm'):
        print(f'Converting {filename} to .nii.gz...')
        filepath, filename = dcm_to_nii(filepath, filename)


    # print status
    print(f'Processing {filename}...')
    processed_filepath = process_volume(filepath, upload_dir)

    # print status
    print(f'Uploading {filename} to MonaiLabel...')

    # Make a request to the MonaiLabel server with the uploaded file
    with open(processed_filepath, 'rb') as f:
        files = {
            'file': (processed_filepath, f, 'application/x-gzip'),
            'params': (None, '{}', 'application/json'),
            'label': (None, '', 'application/json')
        }
        response = requests.post(
            f'http://localhost:8000/infer/{model}?output=image',
            files=files,
        )
    # Save the response content (a .nii.gz file) to a new file
    response_filename = f'segmented_{filename}'
    response_filepath = os.path.join(upload_dir, response_filename)
    with open(response_filepath, 'wb') as f:
        f.write(response.content)

    # Store the response filepath in the session
    session[session_key] = os.path.join(upload_folder, response_filename)

    return redirect(url_for('home'))


@app.route('/infer_pv', methods=['POST'])
def infer_pv():
    return infer_model('pv_livercrop_unet', 'pv_filename')

@app.route('/infer_ap', methods=['POST'])
def infer_ap():
    return infer_model('ha_livercrop_unet', 'ap_filename')

@app.route('/infer_vp', methods=['POST'])
def infer_vp():
    return infer_model('pv_livercrop_unet', 'vp_filename')

@app.route('/download/<path:filepath>')
def download_file(filepath):
    # Return the .nii.gz file as an attachment
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filepath), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

