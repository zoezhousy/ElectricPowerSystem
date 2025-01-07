import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15]
y =[190
,540
,710
,720
,720
,720
,720
,720
,720
,720
,720
,720
,720
,720
,720]

# 创建图表
plt.plot(x, y)  # 使用'o'标记每个数据点

# 添加标题和标签
plt.title('FO# for 10000 strokes')
plt.xlabel('Index')
plt.ylabel('FO')

# 显示图表
plt.show()