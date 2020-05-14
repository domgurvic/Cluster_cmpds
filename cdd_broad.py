#!/usr/bin/env python
# coding: utf-8

# In[38]:


import sys
sys.path.append("/homes/dgurvic/software/miniconda3/envs/ML_Spark/lib/python3.7/site-packages/")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
from collections import Counter


# The idea here is to include all compound-wise G-neg and G-pos data avilable rather than choosing E.coli and S.auresu as representitives. As one option we can use average value across all avilable G-neg and G-pos pMICs to determine nature of compound's activity.
# 
# To to that, following will have to be done:
#     1. Import all data
#     2. Sort through Species 
#     3. Assign g_neg and g_pos tags to species

# The data comes from CDD SPARK, its curated MIC data for wildtype bacteria.

# In[39]:


data_master=pd.read_csv('cdd_full_g.csv', dtype=np.unicode_)


# In[40]:


data_master.info()
data_master_next=data_master[['SMILES', 'Curated & Transformed MIC Data: pMIC', 'Curated & Transformed MIC Data: Species']]


# In[41]:


data_master


# #### Imported data stats: ~50K non-unique datapoints.
# 
# Next we take a look at columns that we care about in this instance:

# In[42]:


data_master_next = data_master_next.rename(columns={"Curated & Transformed MIC Data: pMIC": "pMIC", "Curated & Transformed MIC Data: Species": "Species"})
data_master_next.head(10)


# In[43]:


# quick pMIC values clean up:

data_master_next = data_master_next[data_master_next.pMIC.notnull()] # Remove NaNs
data_master_next['pMIC'] = data_master_next['pMIC'].map(lambda x: x.lstrip('<>=')) # Remove '<'
data_master_next['pMIC'] = data_master_next['pMIC'].map(lambda x: float(x))


# In[44]:


# Unique species:
data_master_next['Species'].unique()


# We see a mix of G-neg and G-pos species, we want to differenciate between the two. So I collect each species in it's own dictionary process them seperatelly:

# In[45]:


g_neg_strain=['Escherichia coli', 'Klebsiella pneumoniae',
       'Acinetobacter baumannii', 'Pseudomonas aeruginosa', 
        'Yersinia enterocolitica',
       'Burkholderia thailandensis', 'Neisseria gonorrhoeae',
       'Proteus mirabilis', 'Enterobacter cloacae',
       'Stenotrophomonas maltophilia', 'Francisella novicida',
       'Vibrio cholerae', 'Salmonella enterica serovar Typhimurium',
       'Proteus hauseri']

g_pos_strain=['Staphylococcus aureus', 'Streptococcus pneumoniae',
       'Enterococcus faecalis', 'Bacillus subtilis']


# Bacteria are split into their respective classes and now I can process the initial dataset with respect to these.

# In[46]:


#UniqueNames = data.Names.unique()

#create a data frame dictionary to store your data frames
g_neg_dict = {elem : pd.DataFrame for elem in g_neg_strain}

for key in g_neg_dict.keys():
    g_neg_dict[key] = data_master_next[:][data_master_next.Species == key]
    
#create a data frame dictionary to store your data frames
g_pos_dict = {elem : pd.DataFrame for elem in g_pos_strain}

for key in g_pos_dict.keys():
    g_pos_dict[key] = data_master_next[:][data_master_next.Species == key]   
    


# G-neg species and amount non-unique datapoints

# In[47]:


for i in g_neg_dict:
    g_neg_dict[i] = g_neg_dict[i].drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    print(len(g_neg_dict[i]), '-', i)


# G-pos species and amout of non-unique datapoints

# In[48]:


for i in g_pos_dict:
    g_pos_dict[i] = g_pos_dict[i].drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    print(len(g_pos_dict[i]), '-', i)


# The above number of datapoitns are unique in respect to their individual strains but not in respect to all strains. Now I will compbine pMIC values for each of the smiles, there is going to be alot of missing data, lets start with g_pos:

# In[49]:


a = pd.merge(g_pos_dict['Enterococcus faecalis'][['SMILES', 'pMIC']], g_pos_dict['Streptococcus pneumoniae'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer', suffixes=['_E. faecalis', '_S. pneumoniae'])
b = pd.merge(a, g_pos_dict['Staphylococcus aureus'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
c = pd.merge(b, g_pos_dict['Bacillus subtilis'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer', suffixes=['_S. aureus', '_B. subtilis'])
c['mean']=c.mean(axis=1, skipna=True)


# In[50]:


c


# For the above we can see that I took an average over 4 avilable g_pos pMIC values, ignoring the NaNs. That way no data is lost. Further inspection of the data revels that scores of the same compound across multiple organisms is not that different, so average seems like a good function here.  The mean column is what i'll use in next steps. I will now do the same with g_neg:

# In[51]:



a1=pd.merge(g_neg_dict['Escherichia coli'][['SMILES', 'pMIC']], g_neg_dict['Klebsiella pneumoniae'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a2=pd.merge(a1, g_neg_dict['Klebsiella pneumoniae'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a3=pd.merge(a2, g_neg_dict['Acinetobacter baumannii'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a4=pd.merge(a3, g_neg_dict['Pseudomonas aeruginosa'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a5=pd.merge(a4, g_neg_dict['Yersinia enterocolitica'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a6=pd.merge(a5, g_neg_dict['Burkholderia thailandensis'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a7=pd.merge(a6, g_neg_dict['Neisseria gonorrhoeae'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a8=pd.merge(a7, g_neg_dict['Proteus mirabilis'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a9=pd.merge(a8, g_neg_dict['Enterobacter cloacae'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a10=pd.merge(a9, g_neg_dict['Stenotrophomonas maltophilia'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a11=pd.merge(a10, g_neg_dict['Francisella novicida'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a12=pd.merge(a11, g_neg_dict['Vibrio cholerae'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a13=pd.merge(a12, g_neg_dict['Salmonella enterica serovar Typhimurium'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')    
a14=pd.merge(a13, g_neg_dict['Proteus hauseri'][['SMILES', 'pMIC']],  on=['SMILES'], how='outer')
a14['mean']=a14.mean(axis=1, skipna=True)
    
# 9053 - Escherichia coli
# 1803 - Klebsiella pneumoniae
# 645 - Acinetobacter baumannii
# 2218 - Pseudomonas aeruginosa
# 977 - Yersinia enterocolitica
# 494 - Burkholderia thailandensis
# 41 - Neisseria gonorrhoeae
# 14 - Proteus mirabilis
# 13 - Enterobacter cloacae
# 13 - Stenotrophomonas maltophilia
# 4 - Francisella novicida
# 4 - Vibrio cholerae
# 4 - Salmonella enterica serovar Typhimurium
# 4 - Proteus hauseri




# Ugly but it works..


# In[52]:


a14[9064:9080]


# Looking at this slice of the data we see that combining pMIC values across 14 known bacteria and deriving their averages seems to work well. Now we can apply same criteria as before and assign 1 and 0 values (at threshold of 5 pMIC) to compounds.

# In[53]:


# Assign labels to compounds

g_neg_mic=a14[['SMILES', 'mean']]
g_pos_mic=c[['SMILES', 'mean']]


g_neg_mic['active'] = [1 if x>=5 else 0 for x in g_neg_mic['mean'].values]
g_pos_mic['active'] = [1 if x>=5 else 0 for x in g_pos_mic['mean'].values]


# We can look at some statistics for our existing datasets:

# In[55]:


print('G-neg active: {}  G-neg inactive: {}\n'.format(Counter(g_neg_mic['active'])[1], Counter(g_neg_mic['active'])[0]) )
print('G-pos active: {}  G-pos inactive: {}\n'.format(Counter(g_pos_mic['active'])[1], Counter(g_pos_mic['active'])[0]) )


# From here, we can take two paths: 
# 
#     1. Construct a model onyl from G-neg data, (active: 2606  vs inactive: 9049) which is definatelly useful in some sense. It would represent G-neg activity model.
#     2. Same as before construct a mdoel based on intersection of datapoints between G-neg data and G-pos data
# 
# For first option, work here is complete. Lets process data for second option and take a look at stats for that. 

# In[56]:


# leave only active g_neg data:
g_pos_all_active=g_pos_mic[g_pos_mic['active']==1].reset_index(drop=True)


# In[60]:


# Save the two datasets for future use

g_pos_all_active.to_csv('g_pos_all_active.csv', index=False)
g_neg_mic.to_csv('g_neg_all.csv', index=False)


# In[61]:


g_neg_mic


# In[19]:


# combine
fin = pd.merge(g_neg_all_active, g_pos_mic,  on=['SMILES'], how='inner')
fin.head(20)


# In[20]:


# Assign final labels
fin['fin_label']= [1 if x ==1 and y==1 else 0 for x, y in zip(fin['active_x'].values,fin['active_y'].values)]
fin


# In[21]:


print('Permeation active: {}  Permeation inactive: {}\n'.format(Counter(fin['fin_label'])[1], Counter(fin['fin_label'])[0]) )


# In[ ]:




