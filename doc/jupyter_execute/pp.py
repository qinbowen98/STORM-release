#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import io
import os

from storm.pp import Cal_CMatrix
img_path='../Visium_Human_Breast_Cancer/spatial/tissue_hires_image.png'
cmtx = Cal_CMatrix()
cmtx.readFile(img_path)
print(cmtx.get_cmtx())


# In[2]:


from storm.VisiumReader import VisiumReader
from storm.pp import VisiumPreprocesser
Reader=VisiumReader()
Reader.read_all(folder_path="../../Visium_Human_Breast_Cancer",gene_token="../../gene_token_homologs.csv",method="binary",key="symbol")
processer=VisiumPreprocesser(Reader,224)
processer.process_tpl()
processer.round_spot()
processer.generate_grid()
processer.map_tissue()
processer.process_adata()
print(processer.fnl_adata)


# In[3]:


from storm.VisiumReader import VisiumReader
from storm.pp import VisiumPreprocesser
import matplotlib.pyplot as plt

Reader=VisiumReader()
Reader.read_all(folder_path="../../Visium_Human_Breast_Cancer",gene_token="../../gene_token_homologs.csv",method="binary",key="symbol")

processer=VisiumPreprocesser(Reader,224)
_,_,_,_,raw_crop_he,crop_he=processer.process_img()
fig,axs=plt.subplots(1,2)
axs[0].imshow(raw_crop_he)
axs[1].imshow(crop_he)

print(processer.tissue_position_list.head(5),processer.final_tpl.head(5),processer.tissue_grid.head(5))


# In[4]:


from storm.pp import exc_he,Cal_CMatrix
import matplotlib.pyplot as plt
from skimage import io
import os
img_path = '../../hm0477/spatial/tissue_hires_image.png'
cmtx = Cal_CMatrix()
cmtx.readFile(img_path)
stain_matrix =cmtx.get_cmtx()
img = io.imread(img_path)
h_image, e_image = exc_he(img, stain_matrix)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(h_image)
ax[1].set_title('Hematoxylin Image')
ax[2].imshow(e_image)
ax[2].set_title('Eosin Image')
plt.show()


# In[5]:


from skimage import io
import matplotlib.pyplot as plt
from storm.pp import exc_tissue
img = io.imread('../../Visium_Human_Breast_Cancer/spatial/tissue_hires_image.png')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].axis('off')
mask=exc_tissue(img,method='otsu')
axes[1].imshow(mask)
axes[1].axis('off')
plt.show()


# In[6]:


from skimage import io
import matplotlib.pyplot as plt
from storm.pp import white_balance_using_white_point,exc_tissue
img = io.imread('../../hm0477/spatial/tissue_hires_image.png')
mask=exc_tissue(img,method='otsu')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].axis('off')
img_balanced=white_balance_using_white_point(img,mask)
axes[1].imshow(img_balanced)
axes[1].axis('off')
plt.show()

