# Disaster Response Pipeline Project

### Aims
The aim of this project is to direct each message  that are sent during disasters to the relief agency that can provide the quickest assistance.  
We want to build a model to classify these messages based on 36 pre-defined categories including Aid Related, Medical Help, Search And Rescue...
This project will involve the building of a basic ETL and Machine Learning pipeline to facilitate the task.
A message can belong to one or more categories, therefore, it is a multi-label classification.
The data set is provided by Figure Eight containing real messages that were sent during disaster events.

### Project Description
There are three parts in this project:

ETL Pipeline: loads and cleans the data, then stores it in a SQLite database.

ML Pipeline: loads the data from SQLite database, builds a text processing and machine learning pipeline, trains and tunes the model using GridSearchCV, outputs the results on the text set and exports the final model as a pickle file.

Flask Web App : uses Pandas to filter data, uses HTML templates, uses Plotly for visualization.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
