###############################################################
## 2022
## RCALIX
###############################################################

import torch, os
import pandas as pd
from torch import cuda
import dask.dataframe as dd
import glob
import gc

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score 
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from torch.utils.data import Dataset



##############################################################

device = 'cuda' if cuda.is_available() else 'cpu'

##############################################################

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


####################################################

all_files = glob.glob("data/input/fungi_ncbi_csv_split_guild/*.csv")
data = pd.concat((pd.read_csv(f) for f in all_files))

#####################################################


combined_labels = { 

"Algal Parasite":    "Parasite", 
"Animal Parasite":   "Parasite",
"Bryophyte Parasite":"Parasite", 
"Fungal Parasite":   "Parasite", 
"Insect Parasite":   "Parasite", 
"Lichen Parasite":   "Parasite", 
"Plant Parasite":    "Parasite",
"Undefined Parasite":"Parasite", 


"Animal Associated Biotroph":  "Other", 
"Animal Endosymbiont":         "Other", 
"Clavicipitaceous Endophyte":  "Other", 
"Endophyte":                   "Other", 
"Epiphyte":                    "Other", 
"Root Associate Biotroph":     "Other", 
"Nematophagous":               "Other",
"Undefined Symbiotroph":       "Other", 

 
"Animal Pathogen":    "Pathogen", 
"Insect Pathogen":    "Pathogen", 
"Plant Pathogen":     "Pathogen", 
"Plant pathogen":     "Pathogen", 


"Arbuscular Mycorrhizal":    "Mycorrhizal", 
"Ectomycorrhizal":           "Mycorrhizal", 
"Endomycorrhizal":           "Mycorrhizal",
"Ericoid Mycorrhizal":       "Mycorrhizal",
"Orchid Mycorrhizal":        "Mycorrhizal", 


"Chitin Saprotroph":       "Saprotroph", 
"Dung Saprotroph":         "Saprotroph", 
"Litter Saprotroph":       "Saprotroph", 
"Leaf Saprotroph":         "Saprotroph", 
"Plant Saprotroph":        "Saprotroph", 
"Soil Saprotroph":         "Saprotroph", 
"Wood Saprotroph":         "Saprotroph",
"Undefined Saprotroph":    "Saprotroph", 


"Lichenized":"Lichenized"

}

data["category"] = data.category.map(lambda x: combined_labels[x.strip()])

print( data.head() )

#####################################################

data = data.sample(frac=0.99, random_state=42)

data = data.groupby('category').head(1000000)

data = data.sample(frac=0.99, random_state=24)  ## % of whole data set

print(data.head())

print("data loaded ...")

#####################################################

## "NULL",

'''
labels = ["Algal Parasite", "Animal Associated Biotroph", "Animal Endosymbiont", "Animal Parasite", "Animal Pathogen", "Arbuscular Mycorrhizal", "Bryophyte Parasite", "Chitin Saprotroph", "Clavicipitaceous Endophyte", "Dung Saprotroph", "Ectomycorrhizal", "Endomycorrhizal", "Endophyte", "Epiphyte", "Ericoid Mycorrhizal", "Fungal Parasite", "Insect Parasite", "Insect Pathogen", "Leaf Saprotroph", "Lichen Parasite", "Lichenized", "Litter Saprotroph", "Nematophagous",  "Orchid Mycorrhizal", "Plant Parasite", "Plant Pathogen", "Plant pathogen", "Plant Saprotroph", "Root Associate Biotroph", "Soil Saprotroph", "Undefined Parasite", "Undefined Saprotroph", "Undefined Symbiotroph", "Wood Saprotroph"]
'''

labels = ["Parasite", "Other", "Pathogen", "Mycorrhizal", "Saprotroph", "Lichenized"]

NUM_LABELS = len(labels)

id2label={i:l for i,l in enumerate(labels)}
label2id={l:i for i,l in enumerate(labels)}

print( label2id )

data["labels"] = data.category.map(lambda x: label2id[x.strip()])

print( data.head() )



#####################################################

SIZE = data.shape[0]

index_size = int(   SIZE*0.80   )

train_texts =  list(data.text[:index_size])
test_texts  =  list(data.text[index_size:])

train_labels = list(data.labels[:index_size])
test_labels  = list(data.labels[index_size:])


#####################################################

print(   len(train_texts), len(test_texts)   )

#####################################################

print(   data.category.value_counts()  )   ##.plot(kind='pie')   ##, figsize=(8,8))
input()


###################################################################

data.to_csv("data/input/CombinedAllLabels.csv", encoding='utf-8', index=False)


###################################################################

print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>")






