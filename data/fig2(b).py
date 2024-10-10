import matplotlib.pyplot as plt
import numpy as np

# 生成随机数据
methods = ['Pluto', 'PPCG', 'PFT_GPU', 'PFT*_GPU']
data1024= [0.0003767270,0.0000423360,0.0000518700,0.0000320211]  # 随机生成速度提升数据

data2048 =[0.0015172340,0.0000652160,0.0000805068,0.0000289851]
data4096 = [0.0067338900,0.0000455360,0.00011778219,0.0000288538]
data1024s = [data1024[0] / t for t in data1024]
data2048s = [data2048[0] / t for t in data2048]
data4096s = [data4096[0] / t for t in data4096]
gmean = [1,31.283030934515725, 19.853586140632505, 52.381398362051144]

def geometric_mean(lst):
    return np.exp(np.mean(np.log(lst)))

# Calculate geometric mean speedups for each method
geometric_mean_speedups = []
for i in range(1, len(data1024s)):
    speedups = [data1024s[i], data2048s[i], data4096s[i]]
    geometric_mean_speedups.append(geometric_mean(speedups))

print(geometric_mean_speedups)

data = np.array([data1024s,data2048s,data4096s,gmean])

# 组名
group_names = ['1024','2048','4096','gmean']

list = ['Pluto','PPCG','PFT_GPU','PFT*_GPU']
# 柱状图位置
x = np.arange(data.shape[0])

# 柱宽
width = 0.13

# 设置颜色
colors = ['#44045A','#413E85','#30688D','#35B777']

# 画图
fig, ax = plt.subplots(figsize=(6,4))
for i in range(data.shape[1]):
    ax.bar(x + i * width, data[:, i], width, label=f'{list[i]}', color=colors[i % len(colors)]) 
    # 使用余数来循环使用颜色列表

# 设置轴标签
ax.set_xlabel('Problem Size')
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
plt.savefig('polynomalmul.png', dpi=600, bbox_inches='tight')
plt.show()