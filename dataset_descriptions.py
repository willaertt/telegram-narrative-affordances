'''
This script can be used to produce UpSetplots and quantitaive overviews of the snowballed datasets after cleaning and filtering the data
The script assumes that the file 'DATA_PARAMS/filtered_dataset_overview.csv' exists 
'''

#import libraries
import pandas as pd
from upsetplot import from_memberships, UpSet
from matplotlib import pyplot as plt
import os
import pandas as pd
import statistics

#define functions
def make_upsetplot(): 
    '''
    Create an upsetplot, store as png in the 'DATA_PARAMS' folder
    This visualization is based on the UpSet plots by Lex et al., 2014 
    For documentation, see: https://upsetplot.readthedocs.io/en/stable/#lex2014
    '''
    #read file with overview of filtered dataset in DATA_PARAMS folder 
    filtered_dataset = pd.read_csv('DATA_PARAMS/filtered_dataset_overview.csv')

    #make upsetplot for the data 
    channel_combinations_list = []
    for name, group in filtered_dataset.groupby('file'):
        channel_combination = list(group['dataset'])
        channel_combinations_list.append(channel_combination)

    data = [','.join(list) for list in channel_combinations_list]
    channel_combinations = from_memberships(channel_combinations_list, data=data)
    plot = UpSet(channel_combinations, show_counts=True, sort_by = 'cardinality').plot() 
    plt.savefig("DATA_PARAMS/dataset_upsetplot.png", dpi = 500)

def get_message_counts(dataset_paths):
    '''
    For a list of paths to datasets of the form 'OUTPUT/{dataset}', produce an overview of message counts per dataset
    Store the output in 'DATA_PARAMS/message_metrics_overview.csv'
    '''

    #initiate list for dicts to make csv
    message_metrics_dicts = []

    #get metrics
    for dataset_path in dataset_paths:

        #get dataset name
        dataset_name = dataset_path.split('/')[-1]

        #set paths
        channel_metrics_path = os.path.join(dataset_path, 'final_metrics/10')
        channel_metrics_files = os.listdir(channel_metrics_path)

        #read csv's and get message counts 
        message_counts = []
        for channel_metrics_file in channel_metrics_files:
            if not channel_metrics_file.endswith('.csv'):
                continue
            channel_metrics_df = pd.read_csv(os.path.join(channel_metrics_path, channel_metrics_file))
            message_count = len(channel_metrics_df.index)
            message_counts.append(message_count)

        total_message_count = sum(message_counts)
        median_message_count = statistics.median(message_counts)
        mean_message_count = statistics.mean(message_counts)

        message_metrics_dict = {'dataset': dataset_name, 'total_message_count': total_message_count, 'median_message_count': median_message_count, 'mean_message_count': mean_message_count}
        message_metrics_dicts.append(message_metrics_dict)

    #make message metrics dataframe and save
    message_metrics_df = pd.DataFrame(message_metrics_dicts)
    message_metrics_df.to_csv('DATA_PARAMS/message_metrics_overview.csv')

if __name__ == "__main__":
    dataset_paths = [] #add paths to datasets
    make_upsetplot()
    get_message_counts(dataset_paths)