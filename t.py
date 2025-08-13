import numpy as np
import os 
from collections import Counter
import matplotlib.pyplot as plt

count_label = np.array([])
for file in os.listdir('./dset/EFSleep/npz/Fpz'):
    if file.endswith('.npz'):
        data = np.load(os.path.join('./dset/EFSleep/npz/Fpz', file))
        # Process the data as needed
        count_label = np.concatenate((count_label, data['label']))
        print(data['fnirs_max_score_index'])

count_label = dict(Counter(count_label))
print(count_label)
print(np.array(list(count_label.values())) / np.array(list(count_label.values())).sum() * 100)

# os.makedirs('./figs', exist_ok=True)
# data = np.load(os.path.join('./dset/EFSleep/npz/Fpz', "S02D2.npz"))
# print(data['eeg'].shape)
# labels = data['label']
# for i in range(len(data['eeg'][0])):
#     if labels[i] == 3:
#         plt.figure(figsize=(20, 10))
#         plt.plot(data['eeg'][i, :])
#         plt.savefig(f'./figs/S02D2_eeg_{i}.png')
#         plt.close()



# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# # === 全局字体与图像参数 === #
# plt.rcParams['figure.dpi'] = 600.0
# plt.rcParams['figure.figsize'] = [3.3, 2.5]
# plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['STIXGeneral']
# plt.rcParams['font.size'] = 8.0
# plt.rcParams['mathtext.fontset'] = 'stix'

# sns.set_theme(style="white", font_scale=1.0, rc={
#     "axes.titlesize": 8,
#     "axes.labelsize": 8,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "legend.fontsize": 7
# })

# # === 数据定义 === #
# labels = ["Wake", "N1", "N2", "N3", "REM"]

# datasets = [
#     ("EFSleep", 
#     np.array([
#         [4267, 182, 73, 49, 74],
#         [164, 2882, 501, 24, 282],
#         [113, 478, 5987, 428, 202],
#         [148, 59, 710, 6142, 54],
#         [103, 754, 199, 37, 1552]
#     ]),
#     np.array([
#     ["91.9%", "3.9%", "1.6%", "1.1%", "1.6%"],  # W类 (行总和: 4645)
#     ["4.3%", "74.8%", "13.0%", "0.6%", "7.3%"],  # N1类 (行总和: 3853)
#     ["1.6%", "6.6%", "83.1%", "5.9%", "2.8%"],   # N2类 (行总和: 7208)
#     ["2.1%", "0.8%", "10.0%", "86.3%", "0.8%"],  # N3类 (行总和: 7113)
#     ["3.9%", "28.5%", "7.5%", "1.4%", "58.7%"]   # R类 (行总和: 2645)
# ]),
#      "#f8cecc"  # 红色
#     )
# ]

# # === 绘图区域 === #
# fig, ax = plt.subplots(1, 1, figsize=(2.6, 2.6))  # 三列图

# for (name, count, percent, hex_color) in datasets:
#     # 自定义颜色渐变
#     cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", hex_color])
#     norm = PowerNorm(gamma=1.0, vmin=0, vmax=100.0)
    
#     percent_map = np.vectorize(lambda s: float(s.strip('%')))(percent)

#     # 拼接注释文本
#     annotations = np.empty_like(percent, dtype=object)
#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             annotations[i, j] = f"{count[i, j]}\n({percent[i, j]})"

#     sns.heatmap(percent_map, 
#                 annot=annotations, 
#                 fmt='', 
#                 cmap=cmap, 
#                 norm=norm,
#                 xticklabels=labels, 
#                 yticklabels=labels if ax is ax else False, 
#                 cbar=False, ax=ax,
#                 annot_kws={"size": 7,
#                            "fontfamily": "Serif",}, 
#                 linewidths=0.3, 
#                 linecolor='gray')

#     ax.set_title(name, fontsize=8, fontfamily="Serif")
#     ax.set_xticklabels(ax.get_xticklabels(), fontfamily="Serif")
#     ax.set_yticklabels(ax.get_yticklabels(), fontfamily="Serif")
#     ax.set_xlabel("Predicted", fontfamily="Serif")
#     ax.set_ylabel("Actual", fontfamily="Serif")

#     ax.tick_params(axis='y', pad=0.5)
#     ax.tick_params(axis='x', pad=0.5)

# # 精确调整边距
# plt.tight_layout(pad=0.5)  # 整体边距
# fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95)

# # === 保存文件 === #
# plt.savefig("confusion_matrices_gamma_adjusted.png", dpi=600,
#             bbox_inches='tight', pad_inches=0.02)
# plt.savefig("confusion_matrice.pdf",
#             bbox_inches='tight', pad_inches=0.02)