import sys
sys.path.append('../../src/')

from flask import Flask,render_template,request
import torch
from dataset import Scattering
from models import EncDecBaseModel

from torchaudio import load
from torchaudio.transforms import Resample



MODEL_FOLDER='./model/model.pt'
word_list=["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]

app = Flask(__name__)
app.config['MODEL_FOLDER'] = MODEL_FOLDER


device = torch.device('cpu')
model = torch.nn.Sequential( Scattering(),EncDecBaseModel(num_mels=125,num_classes=35,final_filter=128,input_length=1600))#EncDecBaseModel(num_mels=64,num_classes=35,final_filter=128,input_length=1601)
model.load_state_dict(torch.load(MODEL_FOLDER))
model.to(device)
model.eval()


@app.route('/')
def root():
    return render_template('index.html',word_list=word_list)


@app.route('/infer', methods=['POST'])
def infer():
    #Retrieve the file 
    file = request.files['file']
    waveform,sample_rate=load(file)
    
    #transform audio
    downsample=Resample(sample_rate, 16000, dtype=waveform.dtype)(waveform.squeeze())
    
    if(len(downsample)<=16000):
        downsample=torch.cat((downsample,torch.zeros(16000-len(downsample))))
    else:
        downsample=downsample[:16000]

    #predict
    return predict(model=model,audio=downsample,device=device)


def predict(model,audio,device):
    audio=audio.squeeze().to(device)
    return word_list[torch.nn.functional.softmax(model(audio.unsqueeze(0)),dim=-1).squeeze().argmax(dim=-1)]

if __name__ == '__main__':
    app.run()