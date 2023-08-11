from sklearn.manifold import TSNE
#%matplotlib inline
import time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

data1 = np.load('tSNE/tsne_result.npy')
print(data1.shape)
# print(len(data1))
# df = pd.read_csv('2F5_4E10_10E8_final.csv')
# y = df ['Class']
# df['Virus'] = y

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_results = tsne.fit_transform(data1)

np.save("tSNE/tsne_fit.npy",tsne_results)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# df = pd.read_json('./pretrain_data/sample.json')
# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]
# fig, ax = plt.subplots(1, 1, figsize = (20, 20), dpi=300)
# cmap = sns.color_palette("Spectral", as_cmap=True)

# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     palette= cmap,
#     data=df,
#     hue = df['spg_symbol'] ,
#     s = 100,
#     alpha=1,
#     ax=ax,
#     legend = 'full')

# ax.legend_.remove()
# ax.set(
#     xlim=(np.min(tsne_results[:,0])-5,np.max(tsne_results[:,0])+5),
#     ylim=(np.min(tsne_results[:,1])-5,np.max(tsne_results[:,1])+5)
# )

# g = df.groupby('Prop')
# survival_rates = g['Prop'].mean()

# norm = matplotlib.colors.Normalize(vmin=survival_rates.min(), vmax=survival_rates.max())
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.5, pad=0.03)
# label = 'Formation Energy'
# #cbar.ax.set_ylabel(label, rotation=270, labelpad=15, fontsize=18)
# cbar.ax.tick_params(size=0, labelsize=24)

#plt.axis('off')
# plt.savefig('tSNE/tsne_ft.png', bbox_inches='tight')
# plt.close()