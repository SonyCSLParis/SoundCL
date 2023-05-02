import sounddevice as sd
import os

import torch
import torchaudio

from models import EncDecBaseModel
from dataset import MfccTransform

def record(seconds=1,sample_rate=16000):

    while True:

        if input("Press Enter to start recording or q to exit :  ")=='q':
            break

        print(f"\033[91m Recording started for {seconds} seconds. \033[00m")
        # Record the audio clip
        recording = sd.rec(int(seconds * sample_rate),samplerate=sample_rate,channels=1)
        sd.wait()
        print("\033[91m Recording ended. \033[00m")

        yield torch.from_numpy(recording)

def predict(model,audio,device):
    audio=audio.squeeze().to(device)
    pre_process=MfccTransform(sample_rate=16000)
    prediction=torch.nn.functional.softmax(model(pre_process(audio).permute(1,0).unsqueeze(0)),dim=-1).squeeze().argmax(dim=-1)
    labels_names = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]
    return labels_names[prediction]

if __name__=='__main__':
    #Define device
    device=torch.device("cpu")
    #Load pytorch model
    PATH='../models/model.pt'
    model = EncDecBaseModel(num_mels=64,num_classes=35,final_filter=128,input_length=1601)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    model.eval()

    for audio in record():
        print(f"\033[92m Predicted: {predict(model,audio,device)}. \033[00m \n")
