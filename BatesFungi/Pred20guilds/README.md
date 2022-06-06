## Fungi Text sequence classification of text into guilds

* 6 guilds
* BERT base
* fine-tuned for 6 classes
* used files in fungi_ncbi_csv_split_guild


## results so far 

  'Parasite': 0, 'Other': 1, 'Pathogen': 2, 'Mycorrhizal': 3, 'Saprotroph': 4, 'Lichenized': 5


Accuracy: 0.17

[[ 8148   267 13743  5964   526 10994]

 [ 8150   284 13655  6059   515 10640]
 
 [ 8064   259 13806  5949   499 10912]
 
 [ 8094   256 13982  6151   523 10652]
 
 [ 8164   309 13763  5924   544 11141]
 
 [ 7907   259 13396  5851   455 11795]]
 
Precision: 0.173

Recall: 0.171

F1-measure: 0.139

              precision    recall  f1-score   support

           0       0.17      0.21      0.18     39642
           1       0.17      0.01      0.01     39303
           2       0.17      0.35      0.23     39489
           3       0.17      0.16      0.16     39658
           4       0.18      0.01      0.03     39845
           5       0.18      0.30      0.22     39663

    accuracy                           0.17    237600
    
   macro avg       0.17      0.17      0.14    237600
   
weighted avg       0.17      0.17      0.14    237600



results data: www.rcalix.com/research/results.txt
