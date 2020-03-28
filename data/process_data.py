import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load two csv files into a dataframe
    
    INPUTS:
        messages_filepath - the path of the messages file (csv)
        categories_filepath - the path of the categories file (csv)
    RETURNS:
        df - the dataframe with both messages and categories 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='left', on='id')
    
    return df


def clean_data(df):
    """
    Clean the disaster message dataframe. 
    
    INPUTS:
        df - the disaster messages dataframe
    RETURNS:
        df - the cleaned disaster messages dataframe
    """
    # create a dataframe for the category column
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # extract names for columns 
    row = categories.iloc[0,:]
    col_names = row.apply(lambda x: x[:-2])
    
    # rename the columns of categories
    categories.columns = col_names
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # change values 2 of a category to 1
    categories = categories.clip(0,1)

    #cleaning the data
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat((df, categories), axis=1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save the dataframe into a SQLite database.
    
    INPUTS:
        df - dataframe
        database_filename - the path to store the SQLite database
    """
    # connect to the data base
    engine = create_engine('sqlite:///'+database_filename)

    conn = engine.connect()

    # Save dataframe to sql table
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

    conn.close()
    engine.dispose()


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