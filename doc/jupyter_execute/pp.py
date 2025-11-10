#!/usr/bin/env python
# coding: utf-8

# In[1]:


from storm.VisiumReader import VisiumReader
from storm.pp import VisiumPreprocesser
Reader=VisiumReader()
Reader.read_all(folder_path="../Visium_Human_Breast_Cancer",gene_token="../gene_token_homologs.csv",method="binary",key="symbol")
processer=VisiumPreprocesser(Reader,224)
processer.process_tpl()
processer.round_spot()
processer.generate_grid()
processer.map_tissue()
processer.process_adata()
print(processer.fnl_adata)


# In[2]:


from storm.VisiumReader import VisiumReader
from storm.pp import VisiumPreprocesser
import matplotlib.pyplot as plt

Reader=VisiumReader()
Reader.read_all(folder_path="../Visium_Human_Breast_Cancer",gene_token="../gene_token_homologs.csv",method="binary",key="symbol")

processer=VisiumPreprocesser(Reader,224)
_,_,_,_,raw_crop_he,crop_he=processer.process_img()
fig,axs=plt.subplots(1,2)
axs[0].imshow(raw_crop_he)
axs[1].imshow(crop_he)

print(processer.tissue_position_list.head(5),processer.final_tpl.head(5),processer.tissue_grid.head(5))

