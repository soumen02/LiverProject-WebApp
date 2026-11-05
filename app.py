import subprocess
from flask import Flask, redirect, request, jsonify, send_from_directory, render_template, url_for, session
import numpy as np
import requests
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import nibabel as nib


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your current directory
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')

# Make sure that the directory exists; if not, create it
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Each phase has its own MONAI Label app (radiologyPV, radiologyHA, radiologyHV in LiverProject-noGUI),
# so each one runs as its own server
MONAI_PV_URL = os.environ.get('MONAI_PV_URL', 'http://localhost:8000')
MONAI_HA_URL = os.environ.get('MONAI_HA_URL', 'http://localhost:8001')
MONAI_HV_URL = os.environ.get('MONAI_HV_URL', 'http://localhost:8002')

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

    if not np.any(mask_data):
        raise ValueError('No liver found in the scan')

    # Create the bounding box mask
    min_x, max_x, min_y, max_y, min_z, max_z = compute_bounding_box(mask_data)

    # Create new mask filled with zeros
    new_mask_data = np.zeros(mask_data.shape)

    # Fill the box with ones
    new_mask_data[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1] = 1

    # Save the mask
    new_mask = nib.Nifti1Image(new_mask_data, mask.affine, mask.header)
    nib.save(new_mask, output_mask_path)

def compute_bounding_box(mask):
    idx = np.where(mask != 0)
    return min(idx[0]), max(idx[0]), min(idx[1]), max(idx[1]), min(idx[2]), max(idx[2])

def apply_bounding_box(input_volume_path, bounding_box_path, output_volume_path):
    # Blank everything outside the bounding box (same result as fslmaths -mas, without needing FSL)
    volume = nib.load(input_volume_path)
    box = nib.load(bounding_box_path).get_fdata()
    cropped = nib.Nifti1Image(volume.get_fdata() * (box > 0), volume.affine, volume.header)
    nib.save(cropped, output_volume_path)

def process_volume(volume_path, output_dir):
    # Apply TotalSegmentator
    output_folder_name = os.path.join(output_dir, 'TotalSegmentator_output')
    subprocess.run(['TotalSegmentator', '-i', volume_path, '-o', output_folder_name, '--roi_subset', 'liver'], check=True)

    print(f'TotalSegmentator Applied {volume_path}...')

    # Parse into the folder, extract the liver label
    liver_mask_path = os.path.join(output_folder_name, 'liver.nii.gz')

    # Create a box the shape of the liver
    box_mask_path = os.path.join(output_dir, 'box_mask.nii.gz')
    create_bounding_box(liver_mask_path, box_mask_path)

    print(f'Bounding Box Created {volume_path}...')

    # Crop the volume to the liver box
    cropped_volume_path = os.path.join(output_dir, 'cropped_' + os.path.basename(volume_path))
    apply_bounding_box(volume_path, box_mask_path, cropped_volume_path)

    print(f'Volume Cropped {volume_path}...')

    return cropped_volume_path

def dcm_to_nii(dcm_path, dcm_filename):
    nii_filename = os.path.splitext(dcm_filename)[0]
    subprocess.run(['dcm2niix', '-z', 'y', '-f', nii_filename, dcm_path], check=True)

    nii_filename = os.path.splitext(nii_filename)[0] + '.nii.gz'
    nii_path = os.path.join(os.path.dirname(dcm_path), nii_filename)
    
    return nii_path, nii_filename

def infer_model(model, session_key, monai_url):
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
    try:
        if filename.endswith('.dcm'):
            print(f'Converting {filename} to .nii.gz...')
            filepath, filename = dcm_to_nii(filepath, filename)

        # print status
        print(f'Processing {filename}...')
        processed_filepath = process_volume(filepath, upload_dir)
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        return jsonify({'error': f'Could not prepare the scan: {e}'}), 500

    # print status
    print(f'Uploading {filename} to MonaiLabel...')

    # Make a request to the MonaiLabel server with the uploaded file
    with open(processed_filepath, 'rb') as f:
        files = {
            'file': (os.path.basename(processed_filepath), f, 'application/x-gzip'),
            'params': (None, '{}', 'application/json'),
            'label': (None, '', 'application/json')
        }
        try:
            response = requests.post(
                f'{monai_url}/infer/{model}?output=image',
                files=files,
            )
        except requests.ConnectionError:
            return jsonify({'error': f'MONAI Label server is not reachable at {monai_url}'}), 502

    # A failed request sends back an error message, not a segmentation
    if response.status_code != 200:
        return jsonify({'error': f'MONAI Label returned {response.status_code} for {model}'}), 502

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
    return infer_model('pv_segmentation', 'pv_filename', MONAI_PV_URL)

@app.route('/infer_ap', methods=['POST'])
def infer_ap():
    return infer_model('ha_segmentation', 'ap_filename', MONAI_HA_URL)

@app.route('/infer_vp', methods=['POST'])
def infer_vp():
    return infer_model('hv_segmentation', 'vp_filename', MONAI_HV_URL)

@app.route('/download/<path:filepath>')
def download_file(filepath):
    # Return the .nii.gz file as an attachment
    return send_from_directory(app.config['UPLOAD_FOLDER'], filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)), use_reloader=False)

