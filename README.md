# Predicting Prostate Cancer Patients’ Vital Status Outcome
Flemming Wu

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import f_classif, SelectKBest, RFE
from imblearn.over_sampling import SMOTENC
from boruta import BorutaPy
from xgboost import XGBClassifier
```

``` python
pd.set_option('display.notebook_repr_html', False)
```

``` python
clinical_df = pd.read_csv('./prad_msk_stopsack_2021_clinical_data.tsv', sep='\t')
clinical_df.columns = clinical_df.columns.str.replace(' ', '_').str.replace('\'', '').str.lower()
genomic_df = pd.read_csv('./data_cna.txt', sep='\t').set_index('Hugo_Symbol').T.reset_index()

df = pd.merge(clinical_df, genomic_df, how='left', left_on='sample_id', right_on='index').drop(columns=['index'])
```

``` python
df.head()
```

                     study_id patient_id          sample_id   8q_arm  \
    0  prad_msk_stopsack_2021  P-0000140  P-0000140-T01-IM3     Gain   
    1  prad_msk_stopsack_2021  P-0000197  P-0000197-T01-IM3  Neutral   
    2  prad_msk_stopsack_2021  P-0000373  P-0000373-T01-IM3  Neutral   
    3  prad_msk_stopsack_2021  P-0000377  P-0000377-T01-IM3     Gain   
    4  prad_msk_stopsack_2021  P-0000391  P-0000391-T01-IM3  Neutral   

       age_at_diagnosis  age_at_procurement      cancer_type  \
    0              42.6                44.0  Prostate Cancer   
    1              79.6                80.4  Prostate Cancer   
    2              54.9                55.2  Prostate Cancer   
    3              60.0                61.3  Prostate Cancer   
    4              45.5                45.8  Prostate Cancer   

                    cancer_type_detailed disease_extent_at_time_impact_was_sent  \
    0  Prostate Neuroendocrine Carcinoma        Metastatic castration-resistant   
    1            Prostate Adenocarcinoma        Metastatic castration-resistant   
    2            Prostate Adenocarcinoma           Metastatic hormone-sensitive   
    3            Prostate Adenocarcinoma        Metastatic castration-resistant   
    4            Prostate Adenocarcinoma                         Regional nodes   

       fraction_genome_altered  ... PTPRS PTPRD  BRAF FAM175A SDHA  PDPK1 BAP1  \
    0                   0.5462  ...   0.0   0.0   0.0     0.0  0.0    0.0  0.0   
    1                   0.0604  ...   0.0   0.0   0.0     0.0  0.0    0.0  0.0   
    2                   0.0023  ...   0.0   0.0   0.0     0.0  0.0    0.0  0.0   
    3                   0.5102  ...   0.0   0.0   0.0     0.0  0.0    0.0  0.0   
    4                   0.0134  ...   0.0   0.0   0.0     0.0  0.0    0.0  0.0   

      SDHB  SDHD PRKAR1A  
    0  0.0   0.0     0.0  
    1  0.0   0.0     0.0  
    2  0.0   0.0     0.0  
    3  0.0   0.0     0.0  
    4  0.0   0.0     0.0  

    [5 rows x 566 columns]
