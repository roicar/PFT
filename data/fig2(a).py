import matplotlib.pyplot as plt

# 问题规模
x = [1024, 2048,4096]
y_ppcg =[52.136876006441234, 128.5966633954858, 736.6971187631764]
y_pft =[39.94848660111818, 104.17206993694941, 365.51417499178103]
y_pft_star =[64.7113309661442, 289.3403852324125, 1162.6281460327582]

# PPCG 的 GFLOPS
# y_ppcg = [523776/0.0000279040, 2072128/0.0000397440, 8386560/0.0000652160,33546240/0.0000455360]

# # # PFT 的 GFLOPS
#y_pft = [523776/0.0000197309, 2072128/0.0000518700, 8386560/0.0000805068,33546240/0.000091778219]

# # # PFT* 的 GFLOPS
# y_pft_star = [523776/0.0000033564, 2072128/0.0000320211, 8386560/0.0000289851,33546240/0.0000288538]

# print([element / 1000000000 for element in y_ppcg])
print([element / 1000000000 for element in y_pft])
# print([element / 1000000000 for element in y_pft_star])
# 创建图形和子图
fig, ax = plt.subplots(figsize=(6,4))

# 绘制 PPCG 折线，使用圆形标记
ax.plot(x, y_ppcg,  linestyle='--',marker='o', label='PPCG')

# 绘制 PFT 折线，使用三角形标记
ax.plot(x, y_pft,linestyle='-', marker='^', label='PFT_GPU')

# 绘制 PFT* 折线，使用方形标记
ax.plot(x, y_pft_star,linestyle='-.', marker='s', label='PFT*_GPU')
ax.set_xticks(x)
ax.set_xticklabels(x)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 添加图例
ax.legend()

# 添加标题和标签
ax.set_xlabel('Problem Size')
ax.set_ylabel('GFLOPS')

# 显示图形
plt.savefig('gflops.png', dpi=600, bbox_inches='tight')
plt.show()
