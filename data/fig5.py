import matplotlib.pyplot as plt
import numpy as np

# 数据
atax=[0.0702525070,0.08265328407287598 ]
bigc=[0.0787168850,0.08876585960388184]
gesumm=[0.0755094700,0.05549502372741699]
mvt=[0.0826531460,0.0802302360534668]

m2m=[0.0975357450,0.08055257797241211]
m3m=[0.0877894490,0.09621763229370117]

gemm=[0.0791413530,0.06890416145324707]
trmm=[2.6108124256,0.07219457626342773]



star5=[0.1013013320,0.09005427360534668]
star9=[0.0872740040,0.07386612892150879]
box9=[0.0784726470, 0.07387471199035645]
box25=[0.1214499810,0.09341049194335938]

syrk=[0.0798151320,0.06024813652038574]
syr2k=[0.1128455720,0.062174081802368164]
polymul =[0.0746048320,0.05958366394042969]
imra=[0.0878036395,0.05199027061462402]
nat=[0.1054506004,0.0793447494506836]

soai=[0.0964829698,0.07183289527893066 ]

ataxs = [atax[0] / t for t in atax]
bigcs = [bigc[0] / t for t in bigc]
gesumms = [gesumm[0] / t for t in gesumm]
mvts=[mvt[0] / t for t in mvt]
m2ms= [m2m[0] / t for t in m2m]
m3ms= [m3m[0] / t for t in m3m]


gemms=[gemm[0] / t for t in gemm]
trmms = [trmm[0] / t for t in trmm]

syrks = [syrk[0] / t for t in syrk]
syr2ks = [syr2k[0] / t for t in syr2k]
polymuls = [polymul[0] / t for t in polymul]
imras = [imra[0] / t for t in imra]
nats=[nat[0] / t for t in nat]
soais= [soai[0] / t for t in soai]
star5s=[star5[0]/ t for t in star5]
star9s=[star9[0]/ t for t in star9]
box9s=[box9[0]/ t for t in box9]
box25s=[box25[0]/ t for t in box25]
geometric_mean = [1,1.197319764393146]
datas = {
    "atax": [0.0702525070, 0.08265328407287598],
    "bigc": [0.0787168850, 0.08876585960388184],
    "gesumm": [0.0755094700, 0.05549502372741699],
    "mvt": [0.0826531460, 0.0802302360534668],
    "m2m": [0.0975357450, 0.08055257797241211],
    "m3m": [0.0877894490, 0.09621763229370117],
    "gemm": [0.0791413530, 0.06890416145324707],
    "star5": [0.1013013320, 0.09005427360534668],
    "star9": [0.0872740040, 0.07386612892150879],
    "box9": [0.0784726470, 0.07387471199035645],
    "box25": [0.1214499810, 0.09341049194335938],
    "syrk": [0.0798151320, 0.06024813652038574],
    "syr2k": [0.1128455720, 0.062174081802368164],
    "polymul":[0.0746048320,0.05958366394042969],
    "imra": [0.0878036395, 0.05199027061462402],
    "nat": [0.1054506004, 0.0793447494506836],
    "soai": [0.0964829698, 0.07183289527893066]
}

# 计算加速比
ratios = [v[0] / v[1] for v in datas.values()]

# 计算几何平均加速比
geometric_mean_speedup = np.prod(ratios) ** (1 / len(ratios))
print(geometric_mean_speedup)



# 数据
data = np.array([ataxs, bigcs, gesumms, mvts, m2ms, m3ms, gemms, star5s,star9s,box9s,box25s,syrks,syr2ks,polymuls,imras,nats,soais,geometric_mean])


# 组名
group_names = ['atax', 'bicg', 'gsumm', 'mvt', '2mm', '3mm', 'gemm','star5','star9','box9','box25','syrk','syr2k','polymul','IMRA','NAT','SOAI','gmean']
list = ['PPCG','PFT*_GPU']
# 柱状图位置
x = np.arange(data.shape[0])

# 柱宽
width = 0.22

# 设置颜色
colors = ['#30688D','#35B777','#F8E620','#ADD8E6']

# 画图
fig, ax = plt.subplots(figsize=(10,3))
for i in range(data.shape[1]):
    ax.bar(x + i * width, data[:, i], width, label=f'{list[i]}', color=colors[i % len(colors)]) 
    # 使用余数来循环使用颜色列表

# 设置轴标签
ax.set_ylabel('Speedup')
# ax.set_xlabel('Groups')

# 设置对数刻度
#ax.set_yscale('log')

# 设置x轴标签
ax.set_xticks(x + width * data.shape[1] / 2)
ax.set_xticklabels(group_names,fontsize=8)

# ax.grid(axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 添加图例
ax.legend(fontsize=8)
plt.savefig('alltime_compare.png', dpi=600, bbox_inches='tight')
plt.show()