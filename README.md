# DataSifterText 2

<!--ToC-->
  Table of Contents
=================
  * [Workflow](#workflow)
  * [Setup](#setup)
  * [Usage](#usage)
  * [See also](#see-also)
  * [References](#references)
<!--ToC-->
            
            ## Workflow:
            ### Steps for the new implementation
###Step 1

The data we use includes structured numerical/categorical columns and unstructured text

###Step 2

Identify sensitive outcome(s) to protect
 
###Step 3 [Yiming]

LightGBM on the text feature/cell to identify predictive keywords for the sensitive outcomes in Step 2 separately.

###Step 4 [Yitong]

Build a "semantic radius" around each keyword to establish a semi-quantitative distance metric to anchor different levels of obfuscation. For example, the top 10 keywords for the outcome "Marital Status" are: 'wife', 'husband', 'married', 'alone', 'daughter', 'widowed', 'son', 'lives', 'sex', 'she'. Based on semantic meanings, we can extract two "semantic clusters": 1: 'wife', 'husband', 'daughter', 'son'; 2: 'married', 'alone', 'widowed'. This step is to guarantee the readability of obfuscated text.
Input 1 (for obfuscation): # keywords (higher = more obfuscation)
Input 2  (for obfuscation): radius around each keyword using word2vec (further away = more obfuscation) 
For example:
small=top 3 keywords, 0-20% distance from each keyword
medium=top 5 keywords, 20-40% distance from each keyword
large=top 10 keywords, 40-60% distance from each keyword
# keyword=k
# words on the semantic radius=N
Extra columns/features for SDV = k + k*N = k*(N+1)

###RECIPES FOR STEPs 4 & 5

???	KEYWORD: wife ??? 'husband', 'married',[0-33%] 'alone', 'daughter', 'widowed',[34-66%] 'son', 'lives', 'sex', 'she'[67-100%]
???	semantic_radius(keyword,#words on the radius=N)
???	word2vec_small(keyword=wife, semantic_radius=search words within dist of 2 from wife,1 to N/3)
???	word2vec_medium(keyword=wife, semantic_radius=search words within dist of 2-4 from wife,N/3 to 2N/3)
???	word2vec_large(keyword=wife, semantic_radius=search words within dist of 4-6 from wife,2N/3 to N)
N words on the spectrum small-medium-large

word2vec: https://en.wikipedia.org/wiki/Word2vec#:~:text=Word2vec%20is%20a%20technique%20for,words%20for%20a%20partial%20sentence. Training on the current data in order to vectorize - DATA DEPENDENT
GloVe: https://en.wikipedia.org/wiki/GloVe_(machine_learning) Training on the current data in order to vectorize - DATA DEPENDENT
BERT: https://en.wikipedia.org/wiki/BERT_(language_model) UNIVERSAL

 
###Step 5

Apply Datasifter to obfuscate Data: swap the keywords within "semantic clusters", may use BERT to further improve readability?
 
### Step 6 [Yitong]

Recast text (before and after obfuscation) into tabular (e.g., incident matrix)
Utility metric: apply LightGBM on the text feature/cell to identify predictive keywords for the sensitive outcomes in Step 2 and after Step 5 (after semantic radius has been ap[plied and keywords swapped).

###Step 7 [Yitong]
Apply SDV Privacy-Utility metrics to the recasted datasets (before and after obfuscation)

###EXAMPLE OF A README FOR THE PSEUDOCODE AND INSTALLATION

            $ cd DataSifterText-package
          ### remove pre-existing env
          $ rm -rf env
          ### define new env
          $ python3 -m venv env
          ### activate virtual env
          $ source env/bin/activate
          ### install required package
          $ pip install -r requirements.txt





            ## Setup:
            ### Set up python virtual environment
            $ cd DataSifterText-package
          ### remove pre-existing env
          $ rm -rf env
          ### define new env
          $ python3 -m venv env
          ### activate virtual env
          $ source env/bin/activate
          ### install required package
          $ pip install -r requirements.txt
          
          ## Usage:
          
          ### Run the whole obfuscation model:
          
          $ python3 total.py <SUMMARIZATION> <KEYWORDS/POSITION SWAP MODE>
            
            SUMMARIZATION 0: no summarize, 1: summarize
          
          KEYWORDS/POSITION SWAP MODE 0: keywords-swap, 1: position-swap
          
          Notice that in summarization mode, we will only do keywords-swap.
          
          ## Example
          $ python total.py 0 0 <filename>
            
            Built-in example:
            python3 total.py 0 0 processed_0_prepare.csv
          
          will run the obfuscation without summarization and doing keywords-swap.
          
          
          ## To train a BERT model:
          ### Clone BERT Github Repository:
          $ git clone https://github.com/google-research/bert.git
          ### Download pre-trained BERT model here (Our work uses BERT-Base, Cased):
          $ https://github.com/google-research/bert#pre-trained-models
          ### Using run_classifier.py in this repository, replace the old run_classifier.py
          ### Create "./data" and "./bert_output" directory
          $ mkdir data
          $ mkdir bert_output
          ### Move train_sifter.py to the directory, run train_sifter.py inside the BERT Repository; make sure the data is in the "./data" directory
          $ cp [your data] data
          $ python3 train_sifter.py
          ### Now the data is ready. run the following command to start training:
          $ python3 run_classifier.py --task_name=cdc --do_train=true --do_eval=true --do_predict=true --data_dir=./data/ --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./cased_L-12_H-768_A-12/bert_config.json --max_seq_length=512 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./bert_output/ --do_lower_case=False
          
          The result will be shown in bert_output directory.
          
          ## See also
          * [DataSifter II (longitudinal Data)](https://github.com/SOCR/DataSifterII)
          * [DataSifter I](https://github.com/SOCR/DataSifter).
          
          ## References
          * [DataSifter-Lite (V 1.0)](https://github.com/SOCR/DataSifter) 
          * [DataSifter website](http://datasifter.org)
          * Marino, S, Zhou, N, Zhao, Yi, Wang, L, Wu Q, and Dinov, ID. (2019) [DataSifter: Statistical Obfuscation of Electronic Health Records and Other Sensitive Datasets](https://doi.org/10.1080/00949655.2018.1545228), Journal of Statistical Computation and Simulation, 89(2): 249-271, DOI: 10.1080/00949655.2018.1545228.
          * Zhou, N, Wang, L, Marino, S, Zhao, Y, Dinov, ID. (2022) [DataSifter II: Partially Synthetic Data Sharing of Sensitive Information Containing Time-varying Correlated Observations](https://journals.sagepub.com/loi/acta), [Journal of Algorithms & Computational Technology](https://journals.sagepub.com/loi/acta), Volume 15: 1-17, DOI: [10.1177/17483026211065379](https://doi.org/10.1177/17483026211065379).
          

