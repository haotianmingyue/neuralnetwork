# 开发者 haotian
# 开发时间: 2021/9/22 14:24
import math

'''
sigmoid激活函数
    w代表权重，w的第一行代表得到第一个输出的权重
    x代表输入的值
    返回一个数组
'''
def sigmoid(w,x):
    t = list()
    for i in range(len(w)):
        n_x :int = 0
        for j in range(len(w[i])):
            n_x +=w[i][j]*x[j]
        t.append(1/(1+math.exp(-n_x)))
    return t
def init_nextwork():
    network = {}
    network['w1'] = [[0.1,0.5,0.4],[0.2,0.3,0.5],[0.3,0.4,0.3]]
    network['w2'] = [[0.3,0.5,0.2],[0.1,0.1,0.8],[0.4,0.4,0.2]]
    #输出一个y
    network['w3'] = [[0.2,0.6,0.2]]
    return network
def forward(network,x):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    a2 = sigmoid(w1,x)
    a3 = sigmoid(w2,a2)
    return sigmoid(w3,a3)

if __name__ == '__main__':

    network = init_nextwork()
    print(forward(network,[0.1,0.2,0.3]))
