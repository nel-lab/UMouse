# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:57:18 2021

@author: Jake

Script to create and plot a kde using either tsne or umap output

"""
from scipy.stats import gaussian_kde
# Gaussian filter for umap/tsne output

embedding = embedding_all
if "tsne" in locals():
    dim_red_method = "TSNE"
else:
    dim_red_method = "UMAP"

if isinstance(embedding, pd.DataFrame):
    embedding_df = embedding
else:
    embedding_df = pd.DataFrame(data = embedding)

#calculate buffer for borders
deltaX = (max(embedding_df.iloc[:,0]) - min(embedding_df.iloc[:,0]))/10
deltaY = (max(embedding_df.iloc[:,1]) - min(embedding_df.iloc[:,1]))/10

#calculate plotting min,max + buffer
xmin = embedding_df.iloc[:,0].min() - deltaX
xmax = embedding_df.iloc[:,0].max() + deltaX
ymin = embedding_df.iloc[:,1].min() - deltaY
ymax = embedding_df.iloc[:,1].max() + deltaY

#make a mesh grid on which to plot stuff
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

#make useful variables then calculate kde
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([embedding_df.iloc[:,0], embedding_df.iloc[:,1]])

# calculate kde from tsne 2dim data
bw_val = 0.15
kernel = gaussian_kde(values, bw_method = bw_val)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
plt.imshow(f)
plt.title('umap gaussian kde BW=' + str(bw_val) + '. DS='+str(downsamp))

# visualize kde plot

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title(dim_red_method + ': 2D Gaussian KDE. BW=' + str(bw_val) + '. DS='+str(downsamp))

