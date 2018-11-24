# Disaster Message Response

## About
This project is an ETL and Machine Learning pipeline on text message data in a disaster scenario. The goal is to correctly classify messages in the event of a disaster in order to reach distressed persons in the population of the disaster. Data provided by Figure Eight.

The message data in the .csv files provided is not balanced. There are very few of some categories. In order to mitigate problems this might create, the model is built to maximize recall (or minimize false negatives). The thought process behind minimizing false negatives is that too much help is better than not enough. 

## Use 
1. Run process_data.py with the following command line arguments (in this order):
  - messages csv file, categories csv file, and filepath of the database in which to save the cleaned data
    - Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
2. Run train_classifier.py with the following command line arguments (in this order):
  - disaster messages database, filepath of pickle file in which to save ML model
    - Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl
3. Ensure engine, df, model and variables in run.py point to desired database, table, and ML pickle file respectively.
4. Run Flask app
  - python run.py
5. Open app to view stats on messages and test label prediction on custom messages

## Files
- ETL Pipeline Preparation is notebook in which the ETL Pipeline is created step by step
- ML Pipeline Preparation is notebook in which the Machine Learning pipeline is created step by step
- Data directory
  - process_data.py cleans data and saves a database
  - messages.csv is a file of text messages received pertaining to a disaster
  - categories.csv is the categorization of the text messages in messages.csv for use in the Machine Learning pipeline
  - database files created by ETL pipeline are also stored here
- Models directory
  - train_classifier.py trains the Machine Learning model on database created in process_data.py and saves model for prediction in flask app
  - Machine Learning models should be saved here
- app directory contains the Flask app which utilizes the developed pipeline and cleaned data for visualizations and predictions of new messages

## At a Glance


## Author
- Original data was provided by Figure Eight. 
- Ben Faus created ETL and ML pipelines (and edited Flask app boiler plate)