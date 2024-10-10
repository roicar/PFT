import matplotlib.pyplot as plt
import numpy as np

# 数据
atax=[0.0012738070,0.0000798720,0.0000572439,0.0000307752 ]
bigc=[0.0017092150,0.0000636160,0.0000443991,0.0000478190]
gesumm=[0.0017643370,0.0000638400,0.000060363942,0.000034797665]
mvt=[0.0011851400,0.0000815360,0.0000512731,0.0000472633 ]

m2m=[2.6668824320,0.0975357450,0.0071800170,0.0009170082]
m3m=[3.7950946170,0.0877894490,0.0259890132,0.0060886795]


gemm=[0.5931656600,0.0028824639,0.003711960992,0.000560372605]
trmm=[0.2559687740,2.6361274719,0.0031172393,0.0004656083]

ataxs = [atax[0] / t for t in atax]
bigcs = [bigc[0] / t for t in bigc]
gesumms = [gesumm[0] / t for t in gesumm]
mvts=[mvt[0] / t for t in mvt]
m2ms= [m2m[0] / t for t in m2m]
m3ms= [m3m[0] / t for t in m3m]


gemms=[gemm[0] / t for t in gemm]
trmms = [trmm[0] / t for t in trmm]
gmean = [1,15.890504025872161, 67.11934414598795, 193.72980505984873]
def geometric_mean(lst):
    return np.exp(np.mean(np.log(lst)))

# Calculate geometric mean speedups for each method
geometric_mean_speedups = []
for i in range(1, len(ataxs)):
    speedups = [
        ataxs[i], bigcs[i], gesumms[i], mvts[i], 
        m2ms[i], m3ms[i], gemms[i], trmms[i]
    ]
    geometric_mean_speedups.append(geometric_mean(speedups))

print(geometric_mean_speedups)


# 数据
data = np.array([ataxs, bigcs, gesumms, mvts, m2ms, m3ms, gemms, trmms,gmean])

# 组名
group_names = ['atax', 'bicg', 'gsumm', 'mvt', '2mm', '3mm', 'gemm', 'trmm','gmean']

list = ['Pluto','PPCG','PFT_CPU','PFT_GPU']
# 柱状图位置
x = np.arange(data.shape[0])

# 柱宽
width = 0.17

# 设置颜色
colors = ['#44045A','#413E85','#30688D','#35B777']

# 画图
fig, ax = plt.subplots(figsize=(9,3))
for i in range(data.shape[1]):
    ax.bar(x + i * width, data[:, i], width, label=f'{list[i]}', color=colors[i % len(colors)]) 
    # 使用余数来循环使用颜色列表

# 设置轴标签
ax.set_ylabel('Speedup')


# 设置对数刻度
ax.set_yscale('log')

# 设置x轴标签
ax.set_xticks(x + width * data.shape[1] / 2)
ax.set_xticklabels(group_names)

# ax.grid(axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 添加图例
ax.legend(fontsize=8)
plt.savefig('nopadding_kernels.png', dpi=600, bbox_inches='tight')
plt.show()