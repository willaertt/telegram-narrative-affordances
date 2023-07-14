'''
Main data processing pipeline for narrative affordances analysis
Filters data by scraper depth and language, create files with message metrics based on svo-parsing and 'novelty', 'transience' and 'resonance' scores, retains channel with minimally requireds number of messages
'''

#import libraries
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial 
import statistics
from decimal import Decimal
import os
import pandas as pd
import emoji
import re
import fasttext
import shutil

import sys
sys.path.append("./SRC/")
from nlp_functions import * 

#define functions
def flatten(l):
    '''
    helper function used when flattening dataframe
    '''
    return [item for sublist in l for item in sublist]

#functions to classify messages based on novelty score patterns
def classify_message(message_row):
    '''
    Classify messages based on 'novelty', 'transience' and 'resonance' scores
    Note that resonance = novelty - transience
    '''
    
    #do not classify if there are NaN values
    if message_row[['novelty', 'transience', 'resonance']].isnull().values.any():
        return None

    novelty = Decimal(message_row['novelty'])
    transience = Decimal(message_row['transience'])
    resonance = Decimal(message_row['resonance'])

    if resonance == 0:
         #novelty 1, transience 1 = 'isolated'
        if novelty == 1 and transience == 1:
            return 'isolated'
        #novelty equals transience, but not both are not equal 1 = continued
        if novelty != 1 and transience != 1: 
            return 'continued'

    if resonance > 0:
        #completely novel, transience after is not equal to 1 (i.e. it must not be completely fleeting) = 'emerging' 
        if novelty == 1 and transience != 1: 
            return 'emerging'

        #more new than transient = 'continued - emerging'
        if novelty != 1 and transience != 1:
            return 'continued-emerging'

    if resonance < Decimal(0):
        #inverse 'emergent' = 'fading'
        if novelty != 1 and transience == 1:
            return 'fading'

        #more transient than new = 'continued - fading'
        if novelty != 1 and transience != 1:
            return 'continued-fading'


def get_message_novelty_scores(triple_df, colname, window, metric):
    '''
    groups a dataframe of triples by message
    vectorizes the list of tokens in a specified column of a dataframe of triples (e.g. subject lemmas) 
    calculates cosine distances between vectors 

    returns a dataframe with additional columns:
    immediate_novelty = cosine distance between current message vector and the one immediately before that
    novelty = average cosine distance between current message and {window} messages before it
    transiense = average cosine distance between current message and {window} messages after it
    resonance = novelty - transience 
    additionally, we add columns with the overlapping concepts for each message (given the window)

    triple_df = a dataframe of triples per message
    colname = the name of a column (e.g. source, target) in a dataframe of triples
    window_size = number of messages before and after each message to consider for calculating novelty and transiense
    metric = the distance measure to use (e.g. 'cosine_distance')

    Comparison mechanisms are inspired by https://www.pnas.org/doi/suppl/10.1073/pnas.1717729115
    '''    
    docs= []
    message_ids = []
    forwards = []
    message_urls = []

    for name, group in triple_df.groupby('message_id'):
        docs.append(list(group[colname].astype(str))) #convert all docs to string
        forwards.append(list(group['forwarded_from'])[0])
        message_urls.append(list(group['message_url'])[0])
        message_ids.append(name)
    vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x) #we already have tokens, so we initiate CountVectorizer with a dummy tokenizer and preprocessor

    vectors = vectorizer.fit_transform(docs).toarray()

    #build up the vector dataframe 
    message_vector_df = pd.DataFrame()
    message_vector_df['message_id'] = message_ids 
    message_vector_df['message_url'] = message_urls
    message_vector_df['vector'] = [array for array in vectors]
    message_vector_df['doc'] = docs
    message_vector_df['forwarded_from'] = forwards

    #loop over the vectors, calculate cosine distance between each vector and the one immediately preceding it
    cos_dists = []
    for index, vector in enumerate(message_vector_df['vector']): 
        if index == 0:
            cos_dists.append(None)
        else:
            vector1 = vector
            vector0 = message_vector_df['vector'][index-1]
            
            if metric == 'cosine_distance':
                cos_dist = spatial.distance.cosine(vector1, vector0)
                cos_dists.append(cos_dist)

            #to do: other metrics can be added here

    message_vector_df['immediate_novelty'] = cos_dists

    #loop over the vectors and concepts with a moving window, calculate average distances to vectors in {window} messages before and after current message
    
    #get a subset of the dataframe with relevant columns
    channel_metrics_df = message_vector_df[['vector', 'doc']]

    overlapping_concepts_before = []
    overlapping_concepts_after = [] 

    avg_cos_dist_preceding_source = []
    avg_cos_dist_following_source = []


    #loop over the rows of the dataframe with a moving window, get distance scores and overlapping concepts
    for index, row in channel_metrics_df.iterrows():
        
        if index < window or index >= len(channel_metrics_df.index) - window: #do not calculate anything for messages out of window (either before or after)

            avg_cos_dist_preceding_source.append(None)
            avg_cos_dist_following_source.append(None)

            overlapping_concepts_before.append(None)
            overlapping_concepts_after.append(None)

        else:
            #get the scores based on the vectors 
            vector = row['vector'] #current vector 
            
            #get pevious and following distances
            previous_distances = []
            following_distances = []

            vector1 = vector
            previous_vectors = channel_metrics_df['vector'][index-window: index]
            for vector0 in previous_vectors:
                previous_distances.append(spatial.distance.cosine(vector1, vector0))
            avg_cos_dist_preceding_source.append(statistics.mean(previous_distances))

            vector1 = vector 
            following_vectors = channel_metrics_df['vector'][index + 1: index + window + 1]
            for vector0 in following_vectors:
                following_distances.append(spatial.distance.cosine(vector1, vector0))
            avg_cos_dist_following_source.append(statistics.mean(following_distances))

            #get the measures for the concepts (i.e. nouns)
            doc = row['doc'] #curent doc
        
            #get previous concepts 
            previous_concepts = channel_metrics_df['doc'][index-window:index].to_list()
            previous_concepts = [item for sublist in previous_concepts for item in sublist]  #flatten list 

            #get following concepts
            following_concepts = channel_metrics_df['doc'][index + 1: index + window + 1].to_list()
            following_concepts = [item for sublist in following_concepts for item in sublist]  #flatten list

            #overlapping concepts
            previous_concepts_overlap = set(doc) & set(previous_concepts)
            following_concepts_overlap = set(doc) & set(following_concepts)

            #add to lists
            overlapping_concepts_before.append(list(previous_concepts_overlap))
            overlapping_concepts_after.append(list(following_concepts_overlap))

    message_vector_df['novelty'] = avg_cos_dist_preceding_source
    message_vector_df['transience'] = avg_cos_dist_following_source
    message_vector_df['resonance'] = message_vector_df['novelty'] - message_vector_df['transience']

    message_vector_df['overlapping_concepts_before'] = overlapping_concepts_before 
    message_vector_df['overlapping_concepts_after'] = overlapping_concepts_after 

    #classify messages based on these scores
    message_vector_df['type'] = message_vector_df.apply(lambda x: classify_message(x), axis = 1)

    return message_vector_df


def SVO_parse_dataframe(telegram_df, base_filename, output_path):
        
    '''
    helper function: SVO parse telegram dataframe 
    '''

    # Retrieve a list of the triples 
    triples = telegram_df.apply(lambda x: get_SVO_triples(text=x.clean_message_text, 
                                                        nlp=nlp, 
                                                        prep_phrases=False, 
                                                        transform_span = lambda x: x.root, #only retain root lemmas, make lowercase
                                                        attribute={"id":             x.message_id,
                                                                    "message_url": x.message_url,
                                                                    "forwarded_from": x.forwarded_from}),
                                axis=1)
    
    #from series of jsons to regular dataframe; we thus only retain the messages that actually have triples in them 
    triples = flatten(triples.to_list())
    triple_df = pd.DataFrame.from_dict(triples)

    #do not proceed if the there are no triples at all (i.e. there can be text, but no triples)
    if triple_df.empty:
        return True

    #cleanup the 'attribute' field
    triple_df['message_id']     = triple_df['attribute'].apply(lambda x: x.get('id'))
    triple_df['forwarded_from'] = triple_df['attribute'].apply(lambda x: x.get('forwarded_from')) 
    triple_df['message_url'] = triple_df['attribute'].apply(lambda x: x.get('message_url'))
    triple_df.drop('attribute', inplace=True, axis=1)

    #drop rows where source or target POS is PRON (remove triples with pronouns)
    triple_df = triple_df[triple_df.apply(lambda x: x['source'].pos_ != 'PRON' and x['target'].pos_ != 'PRON', axis = 1)].reset_index()
    
    #do not proceed if there are no triples left after removing those with pronouns
    if triple_df.empty:
        return True 

    #return triple source and target as strings
    triple_df['source'] = triple_df['source'].apply(lambda x: x.lemma_.lower())
    triple_df['target'] = triple_df['target'].apply(lambda x: x.lemma_.lower())

    #add a table column with a source-target tuple 
    triple_df['tuple'] = triple_df.apply(lambda x: (x.source, x.target), axis = 1)

    #save the file as csv
    csv_filename = base_filename + '_SVO.csv' 
    triple_df.to_csv(os.path.join(output_path, csv_filename), index = False)

    return True

def SVO_parse_file(filename, input_path, output_path, files_in_english):

    '''
    SVO parse individual file
    '''

    #check if the file is in the raw_files_at_depth list (with filtered data)
    if not filename in files_in_english:
        return True #do nothing

    #specify the base filename to be used in output
    base_filename = filename.split('.')[0]

    file = os.path.join(input_path,filename)

    if not os.path.isfile(file): 
        return True
    if not file.endswith('.csv'):
        return True
    
    #skip the scraper_log.csv file (which may still be included if max_scraper_depth == None)
    if filename == 'scraper_log.csv': 
        return True

    print('parse file', filename)

    #load data
    telegram_df = pd.read_csv(filepath_or_buffer= file,
                                sep=',',
                                header=0,
                                usecols=['message_url','clean_message_text', 'forwarded_from'],
                                dtype= str)
    
    #Use message_id as index
    telegram_df['message_id'] = telegram_df.index
    
    #remove posts/rows with emtpy texts
    telegram_df = telegram_df.loc[~telegram_df['clean_message_text'].isnull()]

    #parse
    SVO_parse_dataframe(telegram_df, base_filename, output_path) 
    
    return True
    

def chunker(seq, size):
    '''
    loop over a sequence in chunks of size {chunk}
    '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def svo_parse_filtered_files(project_path, files_in_english):

    '''
    Take the filtered channel data files and add 'source' and target' word columns based on the SVO-parser 
    These processed files are stored in the SVO_parsed_data subfolder of the project folder
    '''

    #take raw data subfolder of the project as input path 
    input_path = os.path.join(project_path, 'raw_data')

    #create an output folder if it does not yet exist 
    output_path = os.path.join(project_path, 'cleaned_triples')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #multiprocess the list of files 
    files = os.listdir(input_path)
    arguments = [[file, input_path, output_path, files_in_english] for file in files]
    pool = Pool()
    _ = pool.starmap(SVO_parse_file, arguments)
    pool.close()

    return True

def calculate_channel_metrics_file(input_path, filename, window, svo_input, metric, output_path, message_treshold):
    '''
    process file (calculate metrics)
    '''

    file = os.path.join(input_path,filename)

    if not os.path.isfile(file): 
        return True
    if not file.endswith('.csv'):
        return True
    
    base_filename = filename.split('.')[0]

    print('calculate for filename' , filename)
    triple_df = pd.read_csv(os.path.join(input_path, filename)) 
    

    #create dataframe with novelty scores for svo columns in the original dataframe, store as csv 
    for svo_label in svo_input:
        if svo_label in ['source', 'target', 'tuple']:

            novelty_scores_df = get_message_novelty_scores(triple_df,svo_label, window, metric)
            novelty_scores_df.to_csv(os.path.join(output_path, base_filename + '.csv'), index = False)

        if svo_label == 'combined_source_target':
            #combined target and create a dataframe that combines the source and target columns into one combined column, store as csv
            source_target_dfs = []

            source_df = triple_df[['source', 'message_id', 'message_url', 'forwarded_from']]
            source_df.columns = ['combined_source_target', 'message_id', 'message_url', 'forwarded_from']
            source_target_dfs.append(source_df)

            target_df = triple_df[['target', 'message_id', 'message_url', 'forwarded_from']]
            target_df.columns = ['combined_source_target', 'message_id', 'message_url', 'forwarded_from']
            source_target_dfs.append(target_df)

            combined_df = pd.concat(source_target_dfs)
        
            source_novelty_scores_df = get_message_novelty_scores(combined_df, svo_label, window, metric)

            #check if the number of remaining triples is above the treshold. If not, do not save. 
            if not len(source_novelty_scores_df.index) >= message_treshold:
                return True

            source_novelty_scores_df.to_csv(os.path.join(output_path, base_filename + '_metrics.csv'))

        return True


def get_channel_metrics_files(project_path, window, svo_input, metric, message_treshold):
    '''
    calculate novelty scores for a directory of SVO-parsed Telegram channel csv files
    returns csv files with scores for 'immediate novelty', 'novelty', 'transience' and 'resonance' per message
    input_path = folder with SVO parsed data (.csv files in project subfolder 'SVO_parsed_data')
    output_path = folder for storing the outputs 
    window = number of messages before and after current one (to be considered when calculating averages)
    col_names = list of columns on which to calculate novelty scores: 'source', 'target', 'tuple', 'combined_source_target'
    '''

    #set input path 
    input_path = os.path.join(project_path, 'cleaned_triples')

    #set output path
    output_path = os.path.join(project_path, 'final_metrics', str(window))

    #make output path if it does not exist yet
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #multiprocess files
    files = os.listdir(input_path)
    arguments = [[input_path, file, window, svo_input, metric, output_path, message_treshold] for file in files]
    pool = Pool()
    _ = pool.starmap(calculate_channel_metrics_file, arguments)
    pool.close()
    return True

def clean_string(text):
    '''
    basic text cleaning function
    takes string, returns string with hyperlinks, mentions, emojis replaced by spaces 
    '''

    #make sure input is string
    text = str(text)
  
    #remove emojis
    text = emoji.get_emoji_regexp().sub(r' ', text)

    #remove @mentions
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)

    #remove hyperlinks
    text = re.sub(r'http\S+', ' ', text)

    #remove multiple spaces
    text = " ".join(text.split())

    #remove leading and trailing spaces
    text = text.strip()
    
    return text 
    

def clean_file_messages(project_path, raw_files_at_depth):
    '''
    add column with cleaned message text to raw data files for selected raw_files_at_depth
    '''
    input_path = os.path.join(project_path, 'raw_data')

    for filename in os.listdir(input_path):

        if not filename in raw_files_at_depth: #skip the messages that are not at required scraper depth
            continue

        file = os.path.join(input_path,filename)

        if not os.path.isfile(file): 
            continue
        if not file.endswith('.csv'):
            continue
        
        #skip the scraper_log.csv file
        if filename == 'scraper_log.csv': 
            continue
        
        message_df = pd.read_csv(file, dtype=str)
        print('get cleaned message text for file', filename)

        message_df['clean_message_text'] = message_df['message_text'].apply(lambda x: clean_string(x))
        message_df.to_csv(file, index = False)

    return True


#fasttext language detection 
def fasttext_detect(msg, ft_model):
    '''
    detect language with fasttext model 
    documentation: https://fasttext.cc/docs/en/language-identification.html 
    '''

    #do not attempt to detect language for empty string
    if msg == None:
        return None

    else: 
        try:
            ln = ft_model.predict(msg)[0][0].split("__")[2] 
        except Exception as e:
            ln = None
        return ln


def language_detect_files(project_path, ft_model, raw_files_at_depth):
    '''
    add column with detected language to raw files
    uses fasttext language detection 
    takes fasttext language model as input
    '''
    input_path = os.path.join(project_path, 'raw_data')

    for filename in os.listdir(input_path):

        if not filename in raw_files_at_depth: #we skip the files that are not in our selection
            continue

        file = os.path.join(input_path,filename)

        if not os.path.isfile(file): 
            continue
        if not file.endswith('.csv'):
            continue
        
        #skip the scraper_log.csv file
        if filename == 'scraper_log.csv': 
            continue

        message_df = pd.read_csv(file, dtype= str)
        print('detect languages for file ', filename)

        message_df['language'] = message_df['clean_message_text'].apply(lambda x: fasttext_detect(x, ft_model))
        message_df.to_csv(file, index = False)

    return True

def filter_data(project_path, channel_filter_list):
    '''
    select scraped channels from 'raw_data' folder based on channel_filter_list, copy them to 'filtered_data' folder
    '''
    raw_data_path = os.path.join(project_path, 'raw_data')
    filtered_data_path = os.path.join(project_path, 'filtered_data')
    if not os.path.exists(filtered_data_path):
        os.makedirs(filtered_data_path)

    for filename in os.listdir(raw_data_path):
        if not filename in channel_filter_list:
            continue
        else:
            print('copy file ', filename)
            source = os.path.join(raw_data_path, filename)
            destination = filtered_data_path
            shutil.copy(source, destination)
    return True 


if __name__ == "__main__":

    #specify project paths (datasets)
    #The project folder should contain a 'raw_data' subfolder with csv outputs of the Telegram scraper, and a 'scraper_log.csv' file with scraper logs
    project_paths = []
    
    #specify treshold for messages in English (with rel frequency of en messages >= language_treshold)
    en_language_treshold = 0.95

    #specify minimum number of messages that should be left after SVO parsing (with message count >= message_treshold)
    message_treshold = 200

    #Specify SVO-parser outputs for which to calculate metrics. 
    #Options are 'source' (only source nodes), 'target' (only target nodes), 'tuple' ((source,target) tuples), 'combined_source_target' (list of combined source and target nodes for message)
    svo_input = ['combined_source_target']

    #Specify distance metric for novelty scores
    metric = 'cosine_distance'

    #Specify window size, i.e. the number of messages before and after to be considered when averaging distance metrics
    window_sizes = [10]

    #initiate a list for channel overview dataframes to be used in upset plot
    final_dataset_overviews = []            

    #go over each of the project paths
    for project_path in project_paths: 
        print('parsing files for ', project_path)

        #first get a list of filenames that are at a completed scraper depth, based on the scraper_log file
        #set paths for raw data and scraper log
        raw_data_path = os.path.join(project_path, 'raw_data')
        scraper_log_file = os.path.join(raw_data_path, 'scraper_log.csv')

        #read scraper log 
        scraper_log_df = pd.read_csv(scraper_log_file)

        #get the list of unique filenames at completed depth from the scraper log
        unique_filename_list = list(set([path_string.split('/')[-1] for path_string in scraper_log_df['file'].dropna()]))
        print('no. of unique filenames in scraper log:', len(unique_filename_list))

        #identify these files in the raw data folder 
        raw_files_at_depth =  []
        for filename in unique_filename_list:
            if filename not in os.listdir(raw_data_path):
                print('file not found in raw data' , filename)
                continue
            raw_files_at_depth.append(filename)
        print('number of these found in raw_data:', len(raw_files_at_depth))

        #enrich selected data with column containing cleaned message texts (remove emojis, hyperlinks, @mentions)
        clean_file_messages(project_path, raw_files_at_depth)

        #add language detection column to raw data files using fasttext (applied to selected cleaned message text). 
        ft_model = fasttext.load_model("SRC/fasttext_lang_detect_models/lid.176.bin") 
        language_detect_files(project_path, ft_model, raw_files_at_depth)

        #get the list of files that have sufficient messages in English
        files_in_english = []
        raw_files_in_english = []
        for filename in raw_files_at_depth:
            raw_file = os.path.join(raw_data_path, filename)
            raw_dataframe = pd.read_csv(raw_file, dtype = str) #set datatype to string
            messages_in_english = raw_dataframe['language'].value_counts(normalize = True)['en']
            if not messages_in_english >= en_language_treshold:
                continue
            files_in_english.append(filename)

        print('number of files in English before filtering for len: ', len(files_in_english))
         
        #extract triples from the filtered data
        svo_parse_filtered_files(project_path, files_in_english)

        #get channel metrics files for the filtered data with extracted triples and minimum message treshold
        for window in window_sizes:
            print('get channel metrics files for windows ', str(window))
            get_channel_metrics_files(project_path, window, svo_input, metric, message_treshold)

        #get a list of the final metrics files
        metrics_path = os.path.join(project_path, 'final_metrics/10')
        metrics_files = os.listdir(metrics_path)

        #store the final metrics files in the combined analysis metrics folder
        print('store combined metrics files')
        combined_metrics_path = "OUTPUT/combined_analysis/final_metrics/10"
        if not os.path.exists(combined_metrics_path):
            os.makedirs(combined_metrics_path)
        for metrics_filename in metrics_files:
            metrics_file = os.path.join(metrics_path, metrics_filename)
            #save as csv
            file_df = pd.read_csv(metrics_file)
            file_df.to_csv(os.path.join(combined_metrics_path, metrics_filename))

        print('store dataset overview')
        #make a dataframe with channel overview to be used for the upset plot, append to list of dataframes
        final_dataset_overview_df  = pd.DataFrame()
        final_dataset_overview_df['file'] = metrics_files
        final_dataset_overview_df['dataset'] = project_path.split('/')[-1]
        final_dataset_overviews.append(final_dataset_overview_df)

    print('make combined dataset')
    #create and store full overview of datasets
    full_dataset_df = pd.concat(final_dataset_overviews)
    full_dataset_df.to_csv(os.path.join("DATA_PARAMS",'filtered_dataset_overview.csv'))