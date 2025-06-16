# app.py
from flask import Flask, render_template, request, send_file, jsonify
import os
from utils import colorize_image, apply_edits
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    colorized_img = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            colorized = colorize_image(filepath)
            colorized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'colorized.png')
            colorized.save(colorized_path)

            colorized_img = 'uploads/colorized.png'
    return render_template('index.html', colorized_img=colorized_img)

@app.route('/live-edit', methods=['POST'])
def live_edit():
    data = request.get_json()
    brightness = float(data.get('brightness', 0))
    contrast = float(data.get('contrast', 1.0))
    border_color = data.get('border_color', '#000000')

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'colorized.png')
    try:
        edited = apply_edits(img_path, brightness, contrast, border_color)
        buf = BytesIO()
        edited.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        print("Edit Error:", e)
        return "Failed to process image", 500

@app.route('/download/colorized')
def download_colorized():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'colorized.png')
    return send_file(path, as_attachment=True)

@app.route('/download/edited')
def download_edited():
    brightness = float(request.args.get('brightness', 0))
    contrast = float(request.args.get('contrast', 1.0))
    border_color = request.args.get('border_color', '#000000')

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'colorized.png')
    edited = apply_edits(img_path, brightness, contrast, border_color)

    buf = BytesIO()
    edited.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name='edited_colorized.png')

if __name__ == '__main__':
    app.run(debug=True)
    