from flask import Flask, redirect, request, jsonify, send_file, render_template, url_for, session
import requests
import os
from werkzeug.utils import secure_filename

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

def infer_model(model, session_key):
    # Check if a file was posted
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    # If the user does not select a file, the browser might
    # submit an empty file without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to the upload folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Make a request to the MonaiLabel server with the uploaded file
    with open(filepath, 'rb') as f:
        files = {
            'file': (filepath, f, 'application/x-gzip'),
            'params': (None, '{}', 'application/json'),
            'label': (None, '', 'application/json')
        }
        response = requests.post(
            f'http://localhost:8000/infer/{model}?output=image',
            files=files,
        )
    # Save the response content (a .nii.gz file) to a new file
    response_filename = f'segmented_{filename}.nii.gz'
    response_filepath = os.path.join(app.config['UPLOAD_FOLDER'], response_filename)
    with open(response_filepath, 'wb') as f:
        f.write(response.content)

    # Store the response filename in the session
    session[session_key] = response_filename

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

@app.route('/download/<filename>')
def download_file(filename):
    # Return the .nii.gz file as an attachment
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
