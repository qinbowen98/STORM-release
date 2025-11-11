#!/usr/bin/env python
# coding: utf-8

# In[1]:


from storm.VisiumReader import VisiumReader

import matplotlib.pyplot as plt
Reader=VisiumReader()
Reader.read_all(folder_path="../Visium_Human_Breast_Cancer",gene_token="../gene_token_homologs.csv",method="binary",key="symbol")
plt.imshow(Reader.raw_he)
print(Reader.adata , Reader.tissue_position_list.head(5) , Reader.scaleJson)


# In[2]:


from storm.VisiumReader import VisiumReader
Reader=VisiumReader()
Reader.read_h5(h5_path="../Visium_Human_Breast_Cancer/filtered_feature_bc_matrix.h5",gene_token="../gene_token_homologs.csv",method="binary",key="symbol")
print(Reader.adata)


# In[3]:


from storm.VisiumReader import VisiumReader

import matplotlib.pyplot as plt
Reader=VisiumReader()
Reader.read_img("../Visium_Human_Breast_Cancer/spatial/tissue_hires_image.png")
plt.imshow(Reader.raw_he)


# In[4]:


from storm.VisiumReader import VisiumReader
Reader=VisiumReader()
base_path="../Visium_Human_Breast_Cancer/spatial"
Reader.read_tissue_position(raw_tpl_path=f"{base_path}/tissue_positions_list.csv",
                            json_path=f"{base_path}/scalefactors_json.json")
print(Reader.tissue_position_list.head(5))

