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
import sklearn

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score 
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from torch.utils.data import Dataset

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

##############################################################

def print_stats_percentage_train_test(y_test, y_pred):    
     print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )

     confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
     print(confmat)
     ## print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

     print('Precision: %.3f' %   precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))
     print('Recall: %.3f' %      recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))
     print('F1-measure: %.3f' %  f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))
     print(classification_report(y_test,   y_pred))


####################################################

'''

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


labels = ["Algal Parasite", "Animal Associated Biotroph", "Animal Endosymbiont", "Animal Parasite", "Animal Pathogen", "Arbuscular Mycorrhizal", "Bryophyte Parasite", "Chitin Saprotroph", "Clavicipitaceous Endophyte", "Dung Saprotroph", "Ectomycorrhizal", "Endomycorrhizal", "Endophyte", "Epiphyte", "Ericoid Mycorrhizal", "Fungal Parasite", "Insect Parasite", "Insect Pathogen", "Leaf Saprotroph", "Lichen Parasite", "Lichenized", "Litter Saprotroph", "Nematophagous",  "Orchid Mycorrhizal", "Plant Parasite", "Plant Pathogen", "Plant pathogen", "Plant Saprotroph", "Root Associate Biotroph", "Soil Saprotroph", "Undefined Parasite", "Undefined Saprotroph", "Undefined Symbiotroph", "Wood Saprotroph"]

'''

#######################################################

data = pd.read_csv('data/input/CombinedAllLabels.csv')
data = data.sample(frac=0.20, random_state=42)                 ## 0.20

#######################################################

labels = ["Parasite", "Other", "Pathogen", "Mycorrhizal", "Saprotroph", "Lichenized"]

NUM_LABELS = len(labels)

id2label={i:l for i,l in enumerate(labels)}
label2id={l:i for i,l in enumerate(labels)}

print( label2id )

#data["labels"] = data.category.map(lambda x: label2id[x.strip()])
#print( data.head() )

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


####################################################

print("loading model from hugginface ...")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", max_length=512)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)

model.to(device)

###################################################

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True)

###################################################

train_dataset = MyDataset(train_encodings, train_labels)
test_dataset  = MyDataset(test_encodings,  test_labels)

###################################################

def compute_metrics(pred): 
    labels = pred.label_ids 
    preds  = pred.predictions.argmax(-1) 
    f1 = f1_score(labels, preds, average='macro') 
    acc = accuracy_score(labels, preds) 
    return {
        'Accuracy': acc,
        'F1': f1
    }

###################################################

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./TTCfungiModel', 
    do_train=True,
    do_eval=False,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=1,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    # Number of steps used for a linear warmup
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="no",
    ## eval_steps=50,
    save_strategy="epoch",
    fp16=True
    #load_best_model_at_end=True
)


#########################################

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics= compute_metrics
)

res_train = trainer.train()

print(   res_train   )

###################################################################

def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    ## return probs, probs.argmax(),model.config.id2label[probs.argmax().item()]
    return model.config.id2label[probs.argmax().item()]

###################################################################

text = "and humans at the seedling stage but negatively correlated at the heading and mature stages. As the most common community members of the plant phyllosphere Methylobacterium plays a critical role in protecting the host plants from various pathogens (Madhaiyan et al. 2006; Ardanov et al. 2012). Alternaria is a strong pathogenic fungus which can cause a variety of plant diseases (Maiti et al. 2007; Logrieco et al. 2009; Kgatle et al. 2018). The relative abundance of Alternaria was positively correlated with elevation which indicated that the higher the elevation the greater the risk of disease in rice. At the heading stage elevation was significantly negatively correlated with Passalora Mycosphaerellaceae_unclassified Periconia etc. while at the mature stage the elevation was significantly positively correlated with Passalora Mycosphaerellaceae_unclassified Periconia and so on indicating that the elevation at the maturity stage and the heading stage had opposing effects on the phyllosphere fungal community. Conclusion In summary using high-throughput sequencing methods this study demonstrated a significant shift in the diversity and community composition of phyllosphere bacteria and fungi at the rice seedling heading and mature stages along an elevational gradient from 580 to 980 m asl. The results showed that the elevation had a greater"

print(    predict(text)      )


####################################################################
# saving the fine tuned model & tokenizer

model_path = "fungi-text-classification-model"

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

###################################################################

list_real_labels = []
list_pred_labels = []

print("predicting on test set ...")

f_out = open('data/output/results.txt', 'w')

performance_test_texts  =  test_texts
performance_test_labels =  test_labels

for i in range(   len(performance_test_texts)    ):
    print('******************************************')
    print(performance_test_texts[i] )
    real_label_string = model.config.id2label[    performance_test_labels[i]   ]
    list_real_labels.append(    performance_test_labels[i]       ) 
    print("real label.......", real_label_string)
    pred_val = predict(performance_test_texts[i] ) 
    print(   pred_val    )
    list_pred_labels.append(   model.config.label2id[pred_val]   )
    f_out.write(   real_label_string + "," + pred_val + "," + performance_test_texts[i] + "\n")
    

f_out.close

####################################################################

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

####################################################################

text_temp = "and humans at the seedling stage but negatively correlated at the heading and mature stages. As the most common community members of the plant phyllosphere Methylobacterium plays a critical role in protecting the host plants from various pathogens (Madhaiyan et al. 2006; Ardanov et al. 2012). Alternaria is a strong pathogenic fungus which can cause a variety of plant diseases (Maiti et al. 2007; Logrieco et al. 2009; Kgatle et al. 2018). The relative abundance of Alternaria was positively correlated with elevation which indicated that the higher the elevation the greater the risk of disease in rice. At the heading stage elevation was significantly negatively correlated with Passalora Mycosphaerellaceae_unclassified Periconia etc. while at the mature stage the elevation was significantly positively correlated with Passalora Mycosphaerellaceae_unclassified Periconia and so on indicating that the elevation at the maturity stage and the heading stage had opposing effects on the phyllosphere fungal community. Conclusion In summary using high-throughput sequencing methods this study demonstrated a significant shift in the diversity and community composition of phyllosphere bacteria and fungi at the rice seedling heading and mature stages along an elevational gradient from 580 to 980 m asl. The results showed that the elevation had a greater"

r2 = nlp(text_temp)
print(r2)

###################################################################
 
print( "  'Parasite': 0, 'Other': 1, 'Pathogen': 2, 'Mycorrhizal': 3, 'Saprotroph': 4, 'Lichenized': 5")
print_stats_percentage_train_test(list_real_labels, list_pred_labels)


###################################################################

print("<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>")






