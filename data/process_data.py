import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    #loads and merges csv files to return a new dataset
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.copy()
    df['categories'] = categories.categories
    return df


def clean_data(df):
    #reformats the category columns and returns a clean dataset for model
    
    categories = df.categories.str.split(';', expand=True)

    row = list(df.categories.str.split(';', expand=True).iloc[0])

    category_colnames = [x[:-2] for x in row]
    print(category_colnames)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)

    categories.columns = category_colnames
    categories.head()
    
    df = df.drop(columns=['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df[df.related != '2']
    return df

def save_data(df, database_filename):
    #saves data into sqlite table for model evaluation
    
    engine = create_engine(database_filename)
    df.to_sql('disastertable', engine, index=False, if_exist='replace')


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
