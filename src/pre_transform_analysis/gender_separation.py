import os
import shutil
from tqdm import tqdm
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch


def load_pickle(file_path: str, mode: str = "rb", encoding=""):
    import pickle

    with open(file_path, mode=mode) as f:
        return pickle.load(f, encoding=encoding)


label2id = load_pickle('./models/label2id.pkl')
id2label = load_pickle('./models/id2label.pkl')

model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=len(label2id), label2id=label2id, id2label=id2label
)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

checkpoint = torch.load('./models/pytorch_model.bin', map_location=torch.device('cpu'))

model.load_state_dict(checkpoint)

# Path to the dataset folder
dataset_path = "../../dataset/SpeechCommands/speech_commands_v0.02"

# Path to the male and female folders
male_folder = "../../dataset/SpeechCommands/MaleCommands/"
female_folder = "../../dataset/SpeechCommands/FemaleCommands/"

# List of supported audio file extensions
audio_extensions = [".wav"]


def classify_voice(input):

    waveform, sr = librosa.load(input)
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
    inputs = feature_extractor(waveform, sampling_rate=feature_extractor.sampling_rate,
                            max_length=16000, truncation=True)
    tensor = torch.tensor(inputs['input_values'][0])
    with torch.no_grad():
        output = model(tensor)
        logits = output['logits'][0]
        label_id = torch.argmax(logits).item()
    label_name = id2label[str(label_id)]
    return label_name


if __name__=='__main__':
    if not os.path.exists(male_folder):
        os.makedirs(male_folder)

    if not os.path.exists(female_folder):
        os.makedirs(female_folder)

    # Traverse the dataset folder and classify audio recordings
    for root, dirs, files in os.walk(dataset_path):
        
        for file in tqdm(files):
            # Check if the file is an audio file
            if any(file.endswith(ext) for ext in audio_extensions):
                audio_file = os.path.join(root, file)
                
                # Classify the voice as male or female
                voice_class = classify_voice(audio_file)
                
                # Create destination folder path based on the voice class
                if voice_class == "male":
                    dest_folder = male_folder + root[len(dataset_path):]
                else:
                    dest_folder = female_folder + root[len(dataset_path):]
                
                # Create the destination folder if it doesn't exist
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                
                # Copy the audio file to the destination folder
                shutil.copy2(audio_file, dest_folder)

    print("Classification and separation completed successfully.")

