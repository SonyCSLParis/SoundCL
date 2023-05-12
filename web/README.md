# Flask Apps: Collect and Infer

This folder contains two Flask apps named Collect and Infer. These apps serve different purposes within a voice-based model development project.

## Collect App

The Collect app is designed to create data sets and save them in a MongoDB database using the Sacred framework. The choice of Sacred was made due to its compatibility with the Dream European project, ensuring integration and compatibility with the project requirements.

The Collect app provides functionalities to collect and manage data sets for training and evaluation of audio-like based models. It allows users to perform the following actions:

- Record data: Users can record samples directly through the app.
- Annotate data: Users can annotate and label the recorded samples to create a labeled data set.
- Store data: The Collect app securely saves the collected data sets in a MongoDB database, ensuring data integrity and easy retrieval for further model training or evaluation.

## Infer App

The Infer app is designed to test the audio-like based models developed in the project. It allows users to record their voice and send the recorded command to the app. The app then runs the data through the model and outputs the corresponding results. The Infer app provides the following functionalities:

- Record data: Users can record their voice commands through the app.
- Model inference: The Infer app utilizes the trained model to process the recorded voice command and generate the desired output or prediction.
- Result display: The app displays the inference results to the user.

Both the Collect and Infer apps are implemented using Flask, a Python web framework.
