import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)

	df = pd.merge(messages, categories, on='id')

	return df



def clean_data(df):
	categories = df['categories'].str.split(';', expand=True)
	row = categories.loc[0]
	
	category_colnames = row.apply(lambda x: x[0:-2])
	categories.columns = category_colnames

	#slice category names from current categories, taking integers off
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].astype(str).str[-1:]
		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])

	#drop old categories and concat new
	df.drop('categories', axis=1, inplace=True)
	df = pd.concat([df,categories],axis=1)

	#drop duplicates
	df = pd.DataFrame(df[df.duplicated()==False])

	#'original' column will not be used. Delete to clean up data.
	df.drop('original', axis=1, inplace=True)

	#eliminate non-binary columns from 'related'
	non_binary_related = df[df['related']==2.].index
	df.drop(non_binary_related, inplace=True)

	return df



def save_data(df, database_filename):
	engine = create_engine('sqlite:///' + str(database_filename))
	df.to_sql('MessageTable', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()