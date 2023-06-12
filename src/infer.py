import sounddevice as sd

import torch

from models import EncDecBaseModel
from dataset import MfccTransform,Scattering

def record(seconds=1,sample_rate=16000):
    """Function to record the user using the machine default microphone 

    Args:
        seconds (int, optional): Defaults to 1.
        sample_rate (int, optional): Defaults to 16000.

    Yields:
        torch.tensor : A 1 dimensional tensor representing the audio
    """

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
    """Function that return the prediction of the model, i.e the argmax of the softmax of the logits.

    Args:
        model (nn.Module): pytorch neural network
        audio (torch.tensor): audio tensor
        device (torch.device): Preferably cpu

    Returns:
        str: predicted label
    """
    audio=audio.squeeze().to(device)
    #pre_process=MfccTransform(sample_rate=16000)
    
    prediction=torch.nn.functional.softmax(model(audio.unsqueeze(0)),dim=-1).squeeze().argmax(dim=-1)
    labels_names = ["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]
    return labels_names[prediction]

if __name__=='__main__':
    """The script enable live testing of the trained models.
    In order to do so choose your model then run the script add follow the instructions to record yourself.
    """
    #Define device
    device=torch.device("cpu")
    #Load pytorch model
    PATH='../models/model.pt'
    model = torch.nn.Sequential( Scattering(),EncDecBaseModel(num_mels=125,num_classes=35,final_filter=128,input_length=1600))#EncDecBaseModel(num_mels=64,num_classes=35,final_filter=128,input_length=1601)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    model.eval()

    for audio in record():
        print(f"\033[92m Predicted: {predict(model,audio,device)}. \033[00m \n")
