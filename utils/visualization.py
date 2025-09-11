import seaborn as sns
import numpy as np

# given a list of labels, assigns a color to each istance, based on its label
def color_palette_mapping(labels):
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette(None, n_clusters)
    
    colors = []
    for l in labels:
        colors.append(palette[l])

    return colors