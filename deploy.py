import os
import requests
from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
from torchvision import transforms, models
import torch
from PIL import Image
import cv2
from tensorflow.keras import models as TF
import numpy as np

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "m4xpl 0it"

def load_model():
    resnet_model = models.resnet50(weights=True)

    for param in resnet_model.parameters():
        param.requires_grad = True

    n_inputs = resnet_model.fc.in_features

    resnet_model.fc = torch.nn.Sequential(torch.nn.Linear(n_inputs, 2048),
                                    torch.nn.SELU(),
                                    torch.nn.Dropout(p=0.4),
                                    torch.nn.Linear(2048, 2048),
                                    torch.nn.SELU(),
                                    torch.nn.Dropout(p=0.4),
                                    torch.nn.Linear(2048, 4),
                                    torch.nn.LogSigmoid())

    for name, child in resnet_model.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = True

    resnet_model.to(DEVICE)

    resnet_model.load_state_dict(torch.load('./assets/bt_resnet50_model.pt', map_location=DEVICE))

    resnet_model.eval()
    return resnet_model

MODEL = load_model()
TF_MODEL = TF.load_model('./assets/kaggle_model.h5')

@app.route('/empty_page')
def empty_page():
    filename = session.get('filename', None)
    os.remove(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('index'))

@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)

def predict(image_path):
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    img = transform(img)

    img = img[None, ...]

    with torch.no_grad():
        y_hat = MODEL.forward(img.to(DEVICE))
        predicted = torch.argmax(y_hat.data, dim=1)
        print("y_hat.data:", y_hat.data)
        print("predicted:", predicted)
        print("Prediction:", LABELS[predicted.data],'\n')
    return {'class_id': str(predicted.data), 'class_name': LABELS[predicted.data]}

def pedict_tensorflow(image_path):
    # the tensorflow model expects BGR image not RGB
    img =cv2.imread(image_path)
    img = cv2.resize(img, (150,150))
    img = img.reshape(1,150,150,3)
    p = TF_MODEL.predict(img)
    p = np.argmax(p,axis=1)[0]

    if p==0:
        p='Glioma'
    elif p==1:
        p="No Tumor"
    elif p==2:
        p='Meningioma'
    else:
        p='Pituitary'

    print('The Model prediction is:', p)
    return p

@app.route('/', methods=['POST', 'GET'])
def index():
    try:
        if request.method == 'POST':
            f = request.files['bt_image']
            filename = str(f.filename)
            print('filename:', filename)
            if filename!='':
                ext = filename.split(".")
                
                if ext[1] in ALLOWED_EXTENSIONS:
                    filename = secure_filename(f.filename)
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(save_path)
                    
                    predicted = predict(save_path)
                    tf_res = pedict_tensorflow(save_path)
                    # session['pred_label'] = f"Pytorch Model: {predicted['class_name']}<br>Tensorflow Model: {tf_res}"
                    session['pred_label'] = [predicted['class_name'], tf_res]
                    session['filename'] = filename
                    return redirect(url_for('pred_page'))
                
    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template('index.html')

if __name__=="__main__":
    app.run(port=8000, debug=True)