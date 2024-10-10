import matplotlib.pyplot as plt
import numpy as np

# 数据
star5=[0.0050610560,0.0000489600,0.0002337940,0.0000357109,0.0000334228,0.0000309321]
star9=[0.0045404260,0.0000585280,0.000446891,0.0000396654,0.0000312503 ,0.0000311539]
box9=[0.0022546069,0.0000535680,0.0002191664,0.0000436966,0.0000313853,0.0000312507]
box25=[0.0074798260,0.0001089920,0.0002154360,0.0001259047,0.0000500768,0.0000497954]

syrk=[0.6352539440,0.0109384321,0.0152551052,0.0130582450,0.0040232151,0.002884425 ]
syr2k=[0.5155599550,0.0428840630,0.0247385971,0.0258755494,0.0073421489,0.007140406]
imra=[0.0007985080,0.0000143360,0.0002533976,0.0000950813,0.0000059354,0.0000071583]
nat=[0.0007442330,0.0000179840,0.0002208117,0.0001056006,0.0000058548,0.0000069489]

soai=[0.0007592210,0.0000245760,0.0000977047,0.0000115140,0.0000049214,0.0000067196]
polymul=[
0.0003767270,0.0000423360,0.000119649140,0.000024910080,0.0000518700,0.0000320211 ]

star5s=[star5[0]/ t for t in star5]
star9s=[star9[0]/ t for t in star9]
box9s=[box9[0]/ t for t in box9]
box25s=[box25[0]/ t for t in box25]
syrks = [syrk[0] / t for t in syrk]
syr2ks = [syr2k[0] / t for t in syr2k]
imras = [imra[0] / t for t in imra]
nats=[nat[0] / t for t in nat]
soais= [soai[0] / t for t in soai]

polymuls=[polymul[0]/t for t in polymul]
gmean =[1,39.90817167965917, 10.588495739358084, 35.108076126258446, 93.33158663266441, 95.82197518800268]
def geometric_mean(lst):
    return np.exp(np.mean(np.log(lst)))

# Calculate geometric mean speedups for each method
geometric_mean_speedups = []
for i in range(1, len(star5s)):
    speedups = [
        star5s[i], star9s[i], box9s[i], box25s[i],syrks[i],syr2ks[i],imras[i],nats[i],soais[i],polymuls[i]
    ]
    geometric_mean_speedups.append(geometric_mean(speedups))

print(geometric_mean_speedups)

# 数据
data = np.array([star5s,star9s,box9s,box25s,syrks,syr2ks,polymuls,imras,nats,soais,gmean])

# 组名
group_names = ['star5','star9','box9','box25','syrk','syr2k','polymul','IMRA','NAT','SOAI','gmean']
list = ['Pluto','PPCG','PFT_CPU','PFT*_CPU','PFT_GPU','PFT*_GPU']
# 柱状图位置
x = np.arange(data.shape[0])

# 柱宽
width = 0.15

# 设置颜色
colors = ['#44045A','#413E85','#30688D','#35B777','#F8E620','#ADD8E6']

# 画图
fig, ax = plt.subplots(figsize=(10,3))
for i in range(data.shape[1]):
    ax.bar(x + i * width, data[:, i], width, label=f'{list[i]}', color=colors[i % len(colors)]) 
    # 使用余数来循环使用颜色列表

# 设置轴标签
ax.set_ylabel('Speedup')
# ax.set_xlabel('Groups')

# 设置对数刻度
ax.set_yscale('log')

# 设置x轴标签
ax.set_xticks(x + width * data.shape[1] / 2)
ax.set_xticklabels(group_names)

# ax.grid(axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 添加图例
ax.legend(fontsize=6)
plt.savefig('padding_kernels.png', dpi=600)
plt.show()