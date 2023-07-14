# telegram-narrative-affordances

This repository contains Python scripts that support the empirical analyses reported on in the research paper "A computational analysis of Telegram's narrative affordances".

The scripts are designed to process CSV files with data from public Telegram channels that can be previewed from the browser. These are to be collected with the dedicated scraper discussed in the paper. Starting from a given seed channel, this scraper will identify related channels based on forwarded messages, and will continue to do so until a specified scraper depth has been achieved, thus implementing a "snowballing" method for collecting Telegram data. For further information, please see the [repository for the scraper](https://github.com/willaertt/telegram-scraper-v1). 

Anonymized outputs for the analyses discussed in the paper are available on [Zenodo](https://doi.org/10.5281/zenodo.8144374).

The remainder of this document provides an overview of each of the Python scripts and their main contributions to the analysis. As a preliminary, it is assumed that each dataset of "snowballed" Telegram channels (CSVs) is stored in its own `OUTPUT/{dataset}/raw_data` folder. The `raw_data` folder should contain a `scraper_log.csv` file output by the scraper, indicating the depths at which a channel was scraped. 

Please note that these scripts are offered for the purposes of transparency only.

---
## 1 Data preprocessing
The script `data_preprocessing.py` comprises the main data peprocessing pipeline. The script filters data by scraper depth and language, creates files with message metrics (based on extracted actants), and retains channels with the required number of messages for further analysis

---
## 2 Dataset descriptions
The script `dataset_descriptions.py` produces UpSetplots and quantitative descriptions such as message counts of the "snowballed" datasets after preprocessing the data. 

---
## 3 Data analysis 
The script `data_analysis.py` performs the paper's main Behavioural Profiles (BP) analysis. 

---
## 4 Overviews of actants
The script `actants_descriptions.py` produces CSVs with the frequencies of detected actants and word clouds with top actants.   

---
## 5 Contact details and license
When using these scripts or parts thereof, please give proper credit by citing the research paper. 
The non-anonymized pipeline that was used to produce the full outputs of the paper resides [in this private repository](https://github.com/willaertt/narratology_affordances).  
Questions or remarks can be sent via [email](mailto:tom.willaert@vub.be). 