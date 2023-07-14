'''
Script for counting actants in datasets, storing actant frequency lists as CSVs, and plotting word clouds of top 200 actants
'''

#import libraries 
from collections import Counter
import pandas as pd
import operator
import ast
import os

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#helper function for getting frequencies
def get_frequencies(list_of_lists):
    '''
    get frequencies for items in list of lists
    return dict with relative frequencies for each item, sorted by 
    '''
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    counts = Counter(flattened_list)
    sorted_counts = dict(sorted(counts.items(),key=operator.itemgetter(1),reverse=True)) #sort 
    return sorted_counts

#helper function for making word clouds
def make_word_cloud(frequency_dict, file_path):
    ''' 
    make a word cloud from a dictionary of the form {'word': frequency, ...} 
    save in file_path
    '''
    wc = WordCloud(
        width = 3200,
        height = 1600,
        background_color = 'white',
        ).generate_from_frequencies(frequency_dict)
    plt.figure(figsize=(20,10))
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(file_path, dpi = 500)
    plt.close()

#actant overview function 
def get_actant_overviews(datasets):
    '''
    Takes a list of dataset names and produces overviews of actants, which are stored in the "DATA_PARAMS/actants" folder 
    '''
    #count actant frequencies for each dataset
    data_path = "OUTPUT"
    output_path = "DATA_PARAMS/actants"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for dataset in datasets:
        print(dataset)
        dataset_path = os.path.join(data_path, dataset + "/final_metrics/10/")
        actant_lists = []
        for filename in os.listdir(dataset_path):
            csv_path = os.path.join(dataset_path, filename)
            metrics_df = pd.read_csv(csv_path, converters = {'doc': ast.literal_eval})
            list_of_actants = list(metrics_df['doc'])
            actant_lists.extend(list_of_actants)

        #get counts for all actants and store as csv 
        actant_counts = get_frequencies(actant_lists)
        actant_df = pd.DataFrame.from_dict(actant_counts, orient = 'index')
        actant_df.index.names = ['actant']
        actant_df.columns = ['count']
        print('save actants as csv')
        actant_df.to_csv(os.path.join(output_path, dataset + '_actants_freqs.csv'))

        #make word cloud for actant frequencies 
        print('make word cloud for actants')
        make_word_cloud(actant_counts, os.path.join(output_path, dataset + '_actants_wc.png'))

if __name__ == "__main__":
    dataset_names = [] #add names of datasets (e.g. 'OUTPUT/{name}')
    get_actant_overviews(dataset_names)