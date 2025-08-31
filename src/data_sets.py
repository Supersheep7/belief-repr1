from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

file_path = "/content/drive/My Drive/Finding_ES_LLM/"

''' True-false (Azariaa & Mitchell 2023) '''

class TrueFalseBuilder():
  def __init__(self):
    self.path = f'{file_path}datasets_diy/true-false'

  def get_dataset(self):
    dfs = []
    for file in tqdm(os.listdir(self.path), desc="Processing files"):
      if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(self.path, file))
        df['filename'] = file
        dfs.append(df)
    df = pd.concat(dfs)
    return df

  def debug(self):
    print(os.listdir(self.path))

''' TruthfulQA (Lin et al. 2022) '''

class TruthfulQABuilder():
  def __init__(self):
    self.path = f'{file_path}datasets_diy/truthfulqa/TruthfulQA.csv'

  def get_dataset(self):
    return pd.read_csv(self.path)

''' Directly from the paper '''

class ITIBuilder():
  def __init__(self, dataset_name):
    if dataset_name == "tqa_mc2":
        self.dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
    elif dataset_name == "tqa_gen":
        self.dataset = load_dataset("truthfulqa/truthful_qa", 'generation')
    elif dataset_name == 'tqa_gen_end_q':
        self.dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
    else:
        raise ValueError("Invalid dataset name")

  def get_dataset(self):

    return self.dataset.to_pandas()

''' MuLan (Fierro et al. 2024) '''

class MuLanBuilder():
  def __init__(self):
    self.json_dataset = load_dataset("coastalcph/fm_queries")

  def get_dataset(self):
    return self.json_dataset['train'].to_pandas()

''' True-false-easy '''

class TrueFalseEasyBuilder():
  def __init__(self, clean=True):
    self.path = f'{file_path}datasets_diy/true-false-easy'
    self.clean = clean

  def get_dataset(self):
    dfs = {}
    df_all = pd.DataFrame()
    to_exclude = ['geonames.csv', 'common_claim.csv', 'likely_old.csv']
    for file in tqdm(os.listdir(self.path), desc="Processing files"):
      if file.endswith('.csv') and file not in to_exclude:
        df = pd.read_csv(os.path.join(self.path, file))
        if self.clean:
          # Drop columns
          if file in ['cities_cities_disj.csv', 'cities_cities_conj.csv']:
            df.drop(columns=['city1', 'city2', 'country1', 'country2', 'correct_country1', 'correct_country2', 'statement1', 'label1', 'statement2', 'label2'], inplace=True)
          elif file in ['cities.csv', 'neg_cities.csv']:
            df.drop(columns=['city', 'country', 'correct_country'], inplace=True)
          elif file in ['larger_than.csv', 'smaller_than.csv']:
            df.drop(columns=['n1', 'n2', 'diff', 'abs_diff'], inplace=True)
          elif file == 'counterfact_true_false.csv':
            df.drop(columns=['relation', 'subject', 'target', 'true_target'], inplace=True)
          elif file == 'likely.csv':
            df.drop(columns=['likelihood'], inplace=True)
        df['filename'] = file
        dfs[file] = df
        df_all = pd.concat([df_all, df])

    print("===================")
    print("WATCH OUT! Datapoints have different column entries depending on the csv")
    print()
    return dfs, df_all

  def debug(self):
    print(os.listdir(self.path))

'''
Following code is adapted from https://github.com/likenneth/honest_llama/tree/master
We avoided tokenizing since I am doing that myself during activation extraction
'''

def get_prompts_tqa(dataset):

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset.iloc[i]['question']
        choices = dataset.iloc[i]['mc2_targets']['choices']
        labels = dataset.iloc[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)):
            choice = choices[j]
            label = labels[j]
            prompt = f"Q: {question} A: {choice}"
            if i == 0 and j == 0:
                print(prompt)
            all_prompts.append(prompt)
            all_labels.append(label)

    return all_prompts, all_labels

def get_prompts_tqa_gen(dataset):

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = dataset.iloc[i]['question']
        category = dataset.iloc[i]['category']

        for j in range(len(dataset.iloc[i]['correct_answers'])):
            answer = dataset.iloc[i]['correct_answers'][j]
            prompt = f"Q: {question} A: {answer}"
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)

        for j in range(len(dataset.iloc[i]['incorrect_answers'])):
            answer = dataset.iloc[i]['incorrect_answers'][j]
            prompt = f"Q: {question} A: {answer}"
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)

    return all_prompts, all_labels, all_categories

def get_prompts_tqa_gen_end_q(dataset):

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = dataset.iloc[i]['question']
        category = dataset.iloc[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset.iloc[rand_idx]['question']

        for j in range(len(dataset.iloc[i]['correct_answers'])):
            answer = dataset.iloc[i]['correct_answers'][j]
            prompt = f"Q: {question} A: {answer} Q: {rand_question}"
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)

        for j in range(len(dataset.iloc[i]['incorrect_answers'])):
            answer = dataset.iloc[i]['incorrect_answers'][j]
            prompt = f"Q: {question} A: {answer} Q: {rand_question}"
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)

    return all_prompts, all_labels, all_categories

def unsupervised_builder(dataset: pd.DataFrame):
    
    X_original = list(dataset['statement'])
    y = list(dataset['label'])
    x0 = []
    x1 = []

    # Generate contrast pairs
    for x in X_original:
      x0.append(f"'{x}' is a true statement")
      x1.append(f"'{x}' is a false statement")        

    return x0, x1, y

def unsupervised_builder_tqa(dataset: pd.DataFrame):
    
    prompts, y, _ = get_prompts_tqa_gen(dataset)
    x0 = []
    x1 = []

    for x in prompts:
      x0.append(f"'{x}'. The answer is correct")
      x1.append(f"'{x}'. The answer is incorrect")

    return x0, x1, y