#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2021/11/3
# @file read_iris.py
def read_iris_data():
    with open('./Data/iris/iris.data','r') as fp:
        iris_data = fp.readlines(0)
    fp.close()
    for i in range(len(iris_data)):
        iris_data[i] = iris_data[i].split(',')
    # print(iris_data)

    iris_x = list()
    iris_y = list()
    for i in range(len(iris_data)):
        t = list()
        for j in range(0,4):
            t.append([float(iris_data[i][j])])
        iris_x.append(t)
        if iris_data[i][-1] == 'Iris-setosa\n':
            iris_y.append([[1],[0],[0]])
        elif iris_data[i][-1] == 'Iris-versicolor\n':
            iris_y.append([[0],[1],[0]])
        elif iris_data[i][-1] == 'Iris-virginica\n':
            iris_y.append([[0],[0],[1]])
    return iris_x,iris_y
