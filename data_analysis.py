'''
Main Behavioural Profiles (BP) analysis pipeline
'''

#import libraries 
from treelib import Tree
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram , linkage
from scipy import cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#data preprocessing helper function
def get_channel_metrics_df(input_path):

    '''
    Takes a directory of channel metrics csvs as input
    Checks if dataframes match min length, filter out dfs with less than {df_min_number_of_posts} posts
    Concatenate dataframes into a single larger dataframe 
    '''
    channel_metrics_dataframes = []
    for filename in os.listdir(input_path): 
        if not filename.endswith('.csv'):
            continue
        else:
            channel_metrics_df = pd.read_csv(os.path.join(input_path, filename), dtype = str)

            #skip files with a type column that contains only empty values (i.e. where all counts will be zero) (technically this should not occur after rough channel filtering)
            if channel_metrics_df['type'].isnull().values.all():
                continue

            channel_metrics_df['filename'] = filename 
            channel_metrics_dataframes.append(channel_metrics_df)

    combined_channels_df = pd.concat(channel_metrics_dataframes)

    return combined_channels_df

def plot_avg_channel_features_per_cluster(target_avg_values, cluster_number, labels, cluster_size, output_path):
    '''
    Make plots of average channel features 
    '''
    y= target_avg_values.values
    x = labels
    fig = plt.figure(figsize = (5,6))
    title = 'Average features cluster '+ str(cluster_number) + '\n' + '(' + str(cluster_size) + ' channels)'
    plt.suptitle(title, fontsize = 18) 
    c = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'black', 'black', 'black', 'black', 'black', 'black']
    plt.bar(x, y, color = c)
    plt.xticks(rotation=90)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.subplots_adjust(bottom=0.45)
    plt.ylim(0, 1)
    filename = 'avg_features_cluster_' + str(cluster_number) + '.png'
    plt.savefig(os.path.join(output_path, filename), dpi = 600)
    plt.close(fig)
    return True 


def make_snake_plots(cluster_df, labels, output_path):
    '''
    Plot the differences in average scores per value in one cluster vs. other clusters.

    Based on Levshina (2015), How to do linguistics with R, section 15.2.4.2 (pp. 313-315).
    We calculate the average values of the features for the target cluster and for the reference clusters.
    For each feature, we substract the average feature value in the reference cluster from that in the target cluster.
    The differences are then plotted. 

    Takes a dataframe of clusters and feature vectors, along with list of feature labels
    Returns images
    '''

    #get difference dataframes
    cluster_sizes = []
    cluster_numbers = []
    diff_dfs = []
    target_avg_value_series = []
    for cluster_number in set(cluster_df['BP_cluster']):

        #store cluster number
        cluster_numbers.append(cluster_number)

        #get target and reference dataframe
        target_df = cluster_df[cluster_df['BP_cluster'] == cluster_number]
        reference_df = cluster_df[cluster_df['BP_cluster'] != cluster_number]

        #store cluster size
        cluster_size = len(target_df.index)
        cluster_sizes.append(cluster_size)

        #calculate the average values in each cluster
        target_avg_values = pd.DataFrame(target_df['vector'].tolist()).mean()
        reference_avg_values = pd.DataFrame(reference_df['vector'].tolist()).mean()

        #store target average values for visulization
        target_avg_value_series.append(target_avg_values)

        #calculate differences between target average values and reference average values, sort in decreasing order
        diff_df = pd.DataFrame()
        diff_df['diff'] = target_avg_values - reference_avg_values
        diff_df['label'] = labels
        diff_df.sort_values('diff', ascending = False).reset_index(inplace = True) #sort df and reset index 

        #append to diff list
        diff_dfs.append(diff_df)

    #factor for spreading out axes of plots
    factor = 1.3 

    #get max x value in all diff_dfs to use across all plots
    max_x_values = []
    max_y_values = []
    for diff_df in diff_dfs:
        max_x_values.append(max([abs(value) for value in diff_df['diff']]))
        max_y_values.append(max([0.3 + item for item in list(diff_df.index)]))

    max_x = max(max_x_values) * factor
    max_y = max(max_y_values) 

    #make figures 
    for (cluster_number, diff_df, cluster_size) in zip(cluster_numbers, diff_dfs, cluster_sizes):

        x = diff_df['diff']
        y = [0.3 + item for item in list(diff_df.index)] #move lowest label slightly above the x-axis to avoid overlap. 

        labels = list(diff_df['label'])

        max_y = max(y) * 1.1 #max_y is assumed to be the same for all clusters

        fig = plt.figure(figsize=(10, 10)) #this was (6,4)
        plt.xlim([-max_x, max_x])
        plt.ylim([0, max_y])

        for (x, y, label) in zip(x, y, labels):
            plt.text(x, y, label, va='center', ha = 'center')

        plt.tick_params(left = False)
        plt.yticks(color='w')

        plt.title('cluster ' + str(cluster_number) + ' <--> other clusters')

        #store plots in output folder
        plot_filename = 'snake_plot_cluster_' + str(cluster_number) + '.png'
        fig.savefig(os.path.join(output_path, plot_filename), dpi = 600) 
        plt.close(fig)

    #plot average feature values, use same max_y_value for all plots
    for target_avg_values, cluster_number, cluster_size in zip(target_avg_value_series, cluster_numbers, cluster_sizes):
        _ = plot_avg_channel_features_per_cluster(target_avg_values, cluster_number, labels, cluster_size, output_path)

    return True

def get_normalized_feature_counts(df, labels):
    '''
    Takes a dataframe of processed Telegram data (metrics) and transforms it into a series of normalized counts for each message type
    df = dataframe with channel metrics 
    labels = labels to be added to the Series (replacing the original required labels)
    '''
    counts = df['type'].value_counts(normalize=True)
    required_types = ['continued', 'continued-emerging', 'continued-fading', 'emerging', 'fading', 'isolated']
    for message_type in required_types:
        if message_type not in counts.index:
            counts.loc[message_type] = float(0)
    #rename index 
    counts = counts.sort_index()
    counts.index = labels
    return counts


def cluster_by_channel_features(channel_metrics_df, output_path, distance_metric, linkage_method, behavioural_profile_type):
    '''
    This pipeline follows the method of 'behavioural profiles' as described in Levchina 2015, ch. 15
    Cluster Telegram channels based on features (i.e. the normalized counts for messages per type (e.g. 'isolated', 'emerging' etc.))
    Aggolomerative clustering is used following Levchina 2015 (ch. 15), the optimal number of clusters is based on silhouette scores 

    channel_metrics_df = a combined dataframe of channel metrics
    output path = a folder for storing plots 
    distance_metric = the distance metric to use during agglomerative clustering (e.g. 'cosine')
    linkage_method = the linkage method used during agglomerative clustering (e.g. 'ward')
    behavioural_profile_type = the composition of the vectors (profiles) to perform clustering on
        * all_proportions = normalized counts for all message types + normalized counts for forwarded message types 
        * forwarded_proportion_diffs = normalized counts for all message types + absolute differences between all message types counts and forwarded message type counts (= observed proportions - expected proportions)

    The function returns:
    a dendogram representation of the clustering
    subplots of the distribution of the features per channel per cluster to help with the interpretation of the clusters
    'snake plots' (Levchina 2015, ch. 15) to help with the interpetation of the clusters
    A visualization of average features per cluster
    '''
    #if the output path does not yet exist, we create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #initiate list of vectors
    vectors = []
    filenames = []

    #loop over channel groups, get vector of proportions of message types
    for filename, group in channel_metrics_df.groupby('filename'):

        #get normalized message type counts of all messages
        all_messages_df = group
        all_messages_counts = get_normalized_feature_counts(all_messages_df, ['all_continued', 'all_continued-emerging', 'all_continued-fading', 'all_emerging', 'all_fading', 'all_isolated'])
     
        #get normalized message type counts of forwarded messages
        forwarded_messages_df = group[~group['forwarded_from'].isnull()]
        forwarded_messages_counts = get_normalized_feature_counts(forwarded_messages_df, ['fw_continued', 'fw_continued-emerging', 'fw_continued-fading', 'fw_emerging', 'fw_fading', 'fw_isolated'])

        #get vector with absolute differences between expected proportion of messages and observed proportions
        expected_fw_proportions = all_messages_counts.values
        observed_fw_proportions = forwarded_messages_counts.values
        abs_fw_diffs = pd.Series(abs(expected_fw_proportions - observed_fw_proportions))
        abs_fw_diffs.index = ['fw_diff_continued', 'fw_diff_continued-emerging', 'fw_diff_continued-fading', 'fw_diff_emerging', 'fw_diff_fading', 'fw_diff_isolated']

        #calculate the proportion of messages that are forwarded messages 
        forwarded_messages_proportion = len(forwarded_messages_df.index) / len(all_messages_df.index)
        forwarded_proportion = pd.Series(forwarded_messages_proportion)
        forwarded_proportion.index = ['forwarded_messages']
        
        #calculate the proportion of regular messages and of forwarded messages that immediately follow the one before (i.e. where the immediate novelty score/distance is not 1.0)
        immediate_novelty_proportion_all = len(all_messages_df[all_messages_df['immediate_novelty'] != 1.0].index)/len(all_messages_df.index)
        if len(forwarded_messages_df.index) == 0:
            immediate_novelty_proportion_forwarded = 0
            immediate_novelty_proportions = pd.Series([immediate_novelty_proportion_all, immediate_novelty_proportion_forwarded])
            immediate_novelty_proportions.index = ['immediately_novel', 'fw_immediately_novel']
        else:
            immediate_novelty_proportion_forwarded = len(forwarded_messages_df[forwarded_messages_df['immediate_novelty'] != 1.0].index)/len(forwarded_messages_df.index)
            immediate_novelty_proportions = pd.Series([immediate_novelty_proportion_all, immediate_novelty_proportion_forwarded])
            immediate_novelty_proportions.index = ['immediately_novel', 'fw_immediately_novel']

        if behavioural_profile_type == 'BP_analysis': #this is the one we use, the ones below were mainly for testing purposes
            combined_series = pd.concat([all_messages_counts, forwarded_messages_counts])

        if behavioural_profile_type == 'BP_analysis_immediate_novelty':
            combined_series = pd.concat([all_messages_counts, forwarded_messages_counts, immediate_novelty_proportions])

        if behavioural_profile_type == 'BP_analysis_all_proportions_and_forwards':
            combined_series = pd.concat([all_messages_counts, forwarded_messages_counts, forwarded_proportion])

        if behavioural_profile_type == 'BP_analysis_forwarded_proportion_diffs':
            combined_series = pd.concat([all_messages_counts, abs_fw_diffs])

        #get vector and append to list
        vector = combined_series.values
        labels = combined_series.index
        vectors.append(vector)

        #append filenames and behavioural profiles series for visualizatons
        filenames.append(filename)

    #compute the matrix of pairwise distances
    k = vectors
    dist_matrix = pdist(np.array(k), distance_metric)

    #clustering (linkage matrix)
    Z = linkage(dist_matrix, method = linkage_method)

    #dendogram visualization
    print('plot dendogram')
    fig = plt.figure(figsize = (5,10))
    dendrogram(Z, labels = filenames, leaf_font_size=8, orientation='right')
    plt.title(distance_metric + '\n' + linkage_method)

    #store dendogram visualization
    plot_filename = 'dendogram.png'
    fig.savefig(os.path.join(output_path, plot_filename), dpi = 500)
    plt.close(fig)

    #calculate silhouette scores to identify optimal number of clusters (for additional information, see e.g. https://twintowertech.com/2020/03/22/automatic-clustering-with-silhouette-analysis-on-agglomerative-hierarchical-clustering/)
    print('calculate silhouette scores and identify optimal number of clusters')
    number_of_observations = len(vectors)
    k = range(2,number_of_observations - 1)
    ac_list = [AgglomerativeClustering(n_clusters = i) for i in k]

    silhouette_scores = {}
    silhouette_scores.fromkeys(k)
    for i,j in enumerate(k):
        silhouette_scores[j] = silhouette_score(vectors,
                            ac_list[i].fit_predict(vectors))

    #get number of clusters that corresponds with the highest silhouette score 
    optimal_number_of_clusters = max(zip(silhouette_scores.values(), silhouette_scores.keys()))[1]
    highest_silhouette_score = max(zip(silhouette_scores.values(), silhouette_scores.keys()))[0]
    print('highest silhouette score', highest_silhouette_score)
    print('optimal number of clusters', optimal_number_of_clusters)

    #store a csv with the metrics for this cluster
    cluster_metrics_dict = {'highest_silhouette_score': highest_silhouette_score, 'optimal_number_of_clusters': optimal_number_of_clusters, 'distance_metric': distance_metric, 'linkage_method': linkage_method}
    pd.DataFrame([cluster_metrics_dict]).to_csv(os.path.join(output_path,'cluster_metrics.csv'))

    #cut the tree at the optimal number of clusters 
    cut_tree = cluster.hierarchy.cut_tree(Z, n_clusters= optimal_number_of_clusters)

    #create a dataframe with filename, series and cluster information 
    cluster_df = pd.DataFrame()
    cluster_df['filename'] = filenames
    cluster_df['BP_cluster'] = [cluster[0] for cluster in cut_tree]
    cluster_df['vector'] = vectors

    #make the snakeplots
    print('make snake plots per cluster and visualize average features per cluster')
    make_snake_plots(cluster_df, labels, output_path)

    return cluster_df


def plot_proportions(array_of_frequencies, output_path, title):
    '''
    create bar chart of proportions 
    input: array of counts per bin
    output path
    title: string with plot title
    returns bar chart 
    '''
    prop_plot_path = os.path.join(output_path, 'proportion_plots')
    if not os.path.exists(prop_plot_path):
        os.makedirs(prop_plot_path)
    fig = plt.figure()
    x = range(len(array_of_frequencies))
    y = array_of_frequencies
    plt.bar(x,y)
    plt.title(title)
    plt.savefig(os.path.join(prop_plot_path, title + '.png'), dpi = 600)
    plt.close(fig)


if __name__ == "__main__":

    project_paths = [] #provide paths of the form "OUTPUT/{dataset_name}"    
    input_paths = [os.path.join(project_path, 'final_metrics/10') for project_path in project_paths]

    #set metrics for behavioural profile clustering
    distance_metric = 'cosine'
    linkage_method = 'ward' 
    behavioural_profile_type = 'BP_analysis' 

    #specify main output path for analysis files (assuming that these are already filtered) 
    output_paths = [os.path.join('/'.join(input_path.split('/')[:-2]), 'analysis') for input_path in input_paths]

    #loop over input folders, 
    for input_path, output_path in zip(input_paths, output_paths):

        #read channel metrics csv files, perform filtering, concatenate into dataframe 
        print('get channel metrics for input path ', input_path)
        channel_metrics_df = get_channel_metrics_df(input_path)
    
        #perform behavioural profile analysis and clustering, store as clustering parameters and metrics as csv, store tree diagram 
        print('perform BP analysis')
        behavioural_profile_analysis_path = os.path.join(output_path, behavioural_profile_type)
        BP_channel_cluster_df = cluster_by_channel_features(channel_metrics_df, behavioural_profile_analysis_path, distance_metric, linkage_method, behavioural_profile_type)
        BP_channel_cluster_df.to_csv(os.path.join(behavioural_profile_analysis_path, 'BP_channel_clusters.csv'))