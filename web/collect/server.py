from sacred import Experiment
from sacred.observers import MongoObserver
from flask import Flask, request, redirect,flash,render_template
from pymongo import MongoClient
import uuid
import glob
import os


UPLOAD_FOLDER = 'temp'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ex=Experiment()
ex.observers.append(MongoObserver(db_name='Recordings'))

@app.route('/')
def root():
    word_list=["backward","bed","bird","cat","dog","down","eight","five","follow","forward","four","go","happy","house","learn","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","visual","wow","yes","zero"]
    return render_template('index.html',word_list=word_list)


@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    _class= request.form['class']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    file_name = str(_class)+"_"+ str(uuid.uuid4()) + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)

    #Run experiment
    ex.add_config({
    'class': _class,
    })
    ex.run_commandline()

    return '<h1>Success</h1>'

@ex.main
def main_database(_run):
    #Add recording as artifact
    list_of_files = glob.glob('./temp/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    _run.add_artifact(latest_file)
    #delete file
    #os.remove(latest_file)


if __name__ == '__main__':
    app.run()