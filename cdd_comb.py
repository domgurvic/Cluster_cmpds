#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/homes/dgurvic/software/miniconda3/envs/ML_Spark/lib/python3.7/site-packages/")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
from collections import Counter



from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
import pandas as pd
import pickle
import os
import joblib
from multiprocessing import Pool
from multiprocessing import Process, Manager, cpu_count
from functools import partial
import random
import psutil
from rdkit import DataStructs
import re
#joblib.parallel_backend('multiprocessing',n_jobs=1)
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
PandasTools.RenderImagesInAllDataFrames(images=True)
from IPython.display import HTML
from rdkit.ML.Cluster import Butina


def calc_sims(fps_1, fps_2):
    sims=[]
    for i in range(0,len(fps_1)):
            #sim = DataStructs.BulkTanimotoSimilarity(fps_1[i], [x for n,x in enumerate(fps_2) if n!= i]) within self
            sim = DataStructs.BulkTanimotoSimilarity(fps_1[i], fps_2) # for two different arrays
            sims.append(sim)
    return sims


def make_canon_smiles(smiles):
    canon_smiles=[]
    for mol in smiles:
        try: 
            canon_smiles.append(Chem.CanonSmiles(mol))
        except:
            return print('Invalid SMILE:', mol)
    return canon_smiles



# The aim here is to combine master MIC data file that has data for g-neg and g-neg data from pMIC file, with g-pos data from pMIC file. 

# In[2]:


pMIC=pd.read_csv('pMIC_2.csv', low_memory=False)
#master_mic=pd.read_csv('master_mic.csv', low_memory=False)


# In[3]:


pMIC.head(50)


# In[4]:


# Split e.coli and s.aureus data

data_aur = pMIC[pMIC['pMIC Data: Organism'] == 'Staphylococcus aureus'].reset_index(drop=True) # S. aureus
data_coli = pMIC[pMIC['pMIC Data: Organism'] == 'Escherichia coli'].reset_index(drop=True) # E. coli

data_aur = data_aur.rename(columns={"pMIC Data: pMIC_Ave": "pMIC", "pMIC Data: Strain": "Strain", "pMIC Data: Active": "Active"})
data_coli = data_coli.rename(columns={"pMIC Data: pMIC_Ave": "pMIC", "pMIC Data: Strain": "Strain", "pMIC Data: Active": "Active"})


# In[5]:


data_aur_all_strains=data_aur.assign(Strain=data_aur.Strain.str.split(';')).explode('Strain') # Expand strains of only S. aureus
#dups_strain = data_2.pivot_table(index=['Strain'], aggfunc='size') # count all strains
#dups_strain.sort_values(ascending=False)


# In[6]:


data_aur_29213_strains = data_aur_all_strains[data_aur_all_strains['Strain'] == 'ATCC 29213'][['SMILES', 'pMIC', 'Active']].reset_index(drop=True)
data_aur_25923_strains = data_aur_all_strains[data_aur_all_strains['Strain'] == 'ATCC 25923'][['SMILES', 'pMIC', 'Active']].reset_index(drop=True)


all_coli = data_coli[['SMILES', 'pMIC', 'Active']]
print(len(all_coli))
all_coli = all_coli.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
print(len(all_coli))


# In[7]:


# Collect only s.auresu ATCC 29213 & 25923 froim all avilable

all_saureus = pd.concat([data_aur_29213_strains, data_aur_25923_strains]).reset_index(drop=True)
print(len(all_saureus))
all_saureus = all_saureus.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
print(len(all_saureus))
Counter(all_saureus['Active'])


# In[8]:


# All g+ data
all_saureus


# In[9]:


# All g- data

Counter(all_coli['Active'])


# In[10]:


# Now assign labels to all g+ and g- and find intersection
# Just g- active:
all_saureus_active =all_saureus[all_saureus['Active']==1].reset_index(drop=True)


fin = pd.merge(all_saureus_active, all_coli,  on=['SMILES'], how='inner', suffixes=['_S. aureus', '_E. coli'])

fin['Class']= [1 if x ==1 and y==1 else 0 for x, y in zip(fin['Active_S. aureus'].values, fin['Active_E. coli'].values)]
#fin = fin.drop(fin[fin['pMIC_x']<5].index).reset_index(drop=True)
fin = fin.dropna().reset_index(drop=True)

Counter(fin['Class'])


# In[11]:


# Compare to oa:
oa_g_neg=pd.read_csv('../spark_data/oa_g-_smiles.csv')
oa_g_pos=pd.read_csv('../spark_data/oa_g+_smiles.csv') # Importt

oa_all = pd.concat([oa_g_neg, oa_g_pos]).reset_index(drop=True)

#oa_smiles= make_canon_smiles(oa_all) # Make them canonical smiles

# Generate molecule objects for oa_mols
oa_mols = [Chem.MolFromSmiles(x) for x in oa_all['Canonical Smiles'].values]
# Generate fingerprints for oa_mols
oa_fps= [AllChem.GetMorganFingerprintAsBitVect(x,3,2048) for x in oa_mols]



#perm_all=pd.read_csv('permeation_source.csv')
perm_mols_fin=[Chem.MolFromSmiles(x) for x in fin['SMILES'].values]
perm_fps_fin= [AllChem.GetMorganFingerprintAsBitVect(x,3,2048) for x in perm_mols_fin]

sim_oa_fin=calc_sims(perm_fps_fin, oa_fps)

dex_fin=[]
for i, compound in enumerate(sim_oa_fin):
    for sim in compound:
        if sim > 0.6:
            if i not in dex_fin:
                dex_fin.append(i)

#print(len(dex_fin))               
fin = fin.drop(dex_fin).reset_index(drop=True)
print('Number of compounds removed de to antibiotic similarity: ', len(dex_fin))
print('pMIC data: Permeation active: {}  Permeation inactive: {}\n'.format(Counter(fin['Class'])[1], Counter(fin['Class'])[0]))


# In[25]:


fin


# In[12]:


#fin[['SMILES', 'Class']].to_csv('permeation_chembl.csv', index=False)


# 881,596
# 
# These are the numbers that I should have gotten in september, but a bunch of inactives slipped through. 
# 
# These are pure E.coli vs S.aureus from Chembl, stringently curated permeation data and compared to antibiotics @ 0.6.
# 
# 
# Now lets combine e.coli data with the rest of MIC data:

# In[13]:


# Import g- and g+ data from master mic

g_pos_active_master = pd.read_csv('g_pos_all_active.csv')
g_pos_active_master = g_pos_active_master.rename(columns={"mean": "pMIC", "active": "Active"})

g_neg_all_master = pd.read_csv('g_neg_all.csv')
g_neg_all_master = g_neg_all_master.rename(columns={"mean": "pMIC", "active": "Active"})


# In[14]:


g_neg_all_master


# In[15]:


# Combine all gpos and gneg:

g_pos_all=pd.concat([g_pos_active_master, all_saureus_active]).reset_index(drop=True)
print(len(g_pos_all))
g_neg_all=pd.concat([g_neg_all_master, all_coli]).reset_index(drop=True)
print(len(g_neg_all))


g_neg_all = g_neg_all.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)

print(len(g_neg_all))
g_pos_all = g_pos_all.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
print(len(g_pos_all))


# In[16]:


print('gram positives: ',Counter(g_pos_all['Active']))


# In[17]:


print('gram negatives: ',Counter(g_neg_all['Active']))


# In[18]:


g_neg_all


# In[49]:


fin_2 = pd.merge(g_pos_all,g_neg_all,  on=['SMILES'], how='inner', suffixes=['_S. aureus', '_E. coli'])
fin_2['Class']= [1 if x ==1 and y==1 else 0 for x, y in zip(fin_2['Active_S. aureus'].values,fin_2['Active_E. coli'].values)]
fin_2 = fin_2.drop(fin_2[fin_2['pMIC_S. aureus']<5].index).reset_index(drop=True)
fin_2=fin_2.dropna().reset_index(drop=True)

print('All data combined: Permeation active: {}  Permeation inactive: {}\n'.format(Counter(fin_2['Class'])[1], Counter(fin_2['Class'])[0]) )


# In[50]:


Counter(fin_2['Class'])


# This is Chembl data combined with every other g- and g+ species on SPARK (obviously disregarding conditions of the test), just taking pMIC values, but! condition issue is somehwat remedied by drawing pMIC line at 5. Now discard all that are at least 0.6 similar similar to current antibiotics

# In[51]:


# Compare compounds to original antibiotics and discard anything above 0.6

#perm_all=pd.read_csv('permeation_source.csv')
perm_mols_fin_2=[Chem.MolFromSmiles(x) for x in fin_2['SMILES'].values]
perm_fps_fin_2= [AllChem.GetMorganFingerprintAsBitVect(x,3,2048) for x in perm_mols_fin_2]

sim_oa_fin_2=calc_sims(perm_fps_fin_2, oa_fps)

dex_fin_2=[]
for i, compound in enumerate(sim_oa_fin_2):
    for sim in compound:
        if sim > 0.6:
            if i not in dex_fin_2:
                dex_fin_2.append(i)

print(len(dex_fin_2))               
fin_2 = fin_2.drop(dex_fin_2).reset_index(drop=True)


# In[52]:


print('All data combined: Permeation active: {}  Permeation inactive: {}\n'.format(Counter(fin_2['Class'])[1], Counter(fin_2['Class'])[0]) )


# And that's the final numbers for all MIC data leveraged for permeation. 

# In[53]:


#fin_2[['SMILES', 'Class']].to_csv('permeation_source.csv', index=False)


# In[54]:


fin_2


# In[55]:


# Now I want to double check that dataset i used before agrees with the newly derived dataset, but prior to that I want check the overlap of two datasets


# In[56]:


fin_3=pd.merge(fin_2[['SMILES', 'Class']], fin[['SMILES', 'Class']],  on=['SMILES'], how='inner', suffixes=['_fin', '_fin_2'])


# In[58]:


fin_3['Class_fin'].values


# In[62]:


Counter((fin_3['Class_fin']==fin_3['Class_fin_2']))


# In[66]:


complete=pd.read_csv('../spark_data/complete.csv')[['SMILES', 'Class']]


# In[73]:


fin_4=pd.merge(complete[['SMILES', 'Class']], fin[['SMILES', 'Class']],  on=['SMILES'], how='inner', suffixes=['_complete', '_fin'])


# In[74]:


fin_4


# In[79]:


(fin_4['Class_complete'].values==fin_4['Class_fin'].values).all()


# In[ ]:




