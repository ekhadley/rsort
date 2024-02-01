import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from colors import *
import csv, math
#np.set_printoptions(precision=4, suppress=True)

def plotvs(table, axis1, axis2):
    fig, ax = plt.subplots()
    ax.plot(table[:, axis1], table[:, axis2], 'bo', markersize=2)
    return fig, ax

def getdata(pth):
    with open(pth, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0], data[1:]

def tabledata(data):
    t = []
    lmap = {"northwest":0, "northeast":1, "southeast":1, "southwest":1}
    ynmap = {"no":0, "yes":1}
    mfmap = {"male": 0, "female": 1}
    for pt in data:
        row = []
        row.append(pt[0]) # age
        row.append(mfmap[pt[1]]) # gender
        row.append(pt[2]) # bmi
        row.append(pt[3]) # num children
        row.append(ynmap[pt[4]]) # smoker
        row.append(lmap[pt[5]]) # location
        row.append(pt[6]) # charges
        t.append(row)
    table = np.float32(t)
    #for col in range(len(table[0])):
    #    table[:, col] /= np.max(table[:, col])
    #    table[:, col] -= np.min(table[:, col])
    for col in range(len(table[0])-1):
        print(f"{bold+underline+blue}column {col}. mean: {np.mean(table[:, col])}, var: {np.var(table[:, col])}{endc}")
        table[:, col] -= np.mean(table[:, col])
        table[:, col] /= np.var(table[:, col])
    costs = table[:, -1]
    costs -= np.mean(table[:, col])
    costs /= math.sqrt(np.var(table[:, col]))
    #print(bold, underline, green, costs.min(), costs.max(), endc)
    #exit()
    return table

def dataset(table, ratio=0.9):
    X, Y = table[:, :-1], table[:, -1]
    trainX, testX = X[:int(len(X)*ratio)], X[int(len(X)*ratio):]
    trainY, testY = Y[:int(len(Y)*ratio)], Y[int(len(Y)*ratio):]
    return trainX, trainY, testX, testY

def predict(m, x):
    return m[0] + np.dot(x, m[1:])

def loss(m, x, y):
    pred = predict(m, x)
    diff = pred - y
    #grad = diff*m[1:]
    #grad = np.concatenate(([diff], grad))
    grad = np.concatenate(([diff], 2*diff*x))
    return pred, diff, grad

def eval(m, X, Y):
    diffs = []
    rels = []
    for x, y in zip(X, Y):
        pred = predict(m, x)
        diff = pred - y
        diffs.append(abs(diff))
        rels.append(abs(pred/y))
    return np.mean(diffs), np.mean(rels)

def update(m, x, y, lr=0.002):
    pred, diff, grad = loss(m, x, y)
    update = lr*grad
    m -= update
    return pred, diff, update

columns, data = getdata("D:\\insurance.csv")
table = tabledata(data)

trainX, trainY, testX, testY = dataset(table)

m = np.random.uniform(-1, 1, len(trainX[0])+1)

LR = 0.01
diffs = []
for s in trange(1):
    i = np.random.randint(len(trainX))
    pred, diff, upd = update(m, trainX[i], trainY[i], lr=LR)
    diffs.append(diff**2)
    print(f"{blue}example: {trainX[i]} = {green}{trainY[i]}")
    print(f"{orange}pred: {pred}. {red}diff: {diff}")
    print(f"{lime}example update size: {upd}")
    print(f"{bold+underline+pink}parameters: {m}{endc}")
    print()

def line(m, b, fig):
    _, maxx = fig.get_xlim()
    _, maxy = fig.get_ylim()
    # m*x + b = maxy => x = (maxy-b)/m
    x = np.linspace(0, max(maxx, (maxy-b)/m), 100)
    return x, m*x+b



d, r = eval(m, trainX, trainY)
print(f"{bold+underline+red}train error: {d}{endc}")
print(f"{bold+underline+purple}train relative error: {r}{endc}")

_, dplot = plt.subplots()
dplot.plot(diffs, 'bo', markersize=2)

_, bmi = plotvs(table, 0, -1)
bmi.plot(*line(m[1], m[0], bmi), color="green")

plt.show()