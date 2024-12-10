from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
import os

from livepredictions import LivePredictions

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

live_prediction = LivePredictions()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            # This handles the case when no file part is present
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            # This handles the case when no file is selected
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predicted_emotion = live_prediction.make_predictions(file=filepath)
            return render_template("result.html", emotion=predicted_emotion, audio_filename=filename)

    # For GET requests, or if no file was uploaded, render index.html
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
