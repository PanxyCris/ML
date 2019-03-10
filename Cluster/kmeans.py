import pandas as pd
import numpy as np
import random


class KMeans:
    D = []
    k = 0
    cluster = []

    def __init__(self, str, k):
        self.load_data(str)
        self.k = k

    def load_data(self, str):
        self.D = pd.read_csv(str)

    def distance(self, x, u):
        d = 0
        for col in self.D.columns.values[1:]:
            d += (x[col] ** 2 - u[col] ** 2)
        return abs(d) ** 0.5

    def getMinIndex(self, data):
        minValue = np.Inf
        minIndex = 0
        for i in range(len(data)):
            if data[i] < minValue:
                minValue = data[i]
                minIndex = i
        return minIndex

    def getNewVectorValue(self, vector):
        attributes = vector[0].index.values[1:]
        u = pd.Series(index=self.D.columns.values)
        for i in attributes:
            u[i] = 0.0
        for i in range(len(vector)):
            x = vector[i]
            for j in attributes:
                u[j] += x[j]
        for i in range(len(u)):
            u[i] /= len(vector)
        return u

    def calculate(self):
        # choose random k data as mean vector
        sample = pd.DataFrame(random.sample(self.D.values.tolist(), self.k), columns=self.D.columns.values)
        m = self.D.shape[0]
        isUpdate = True
        while isUpdate:
            isUpdate = False
            C = [[] for i in range(self.k)]
            d = pd.DataFrame([[np.Inf for i in range(self.k)] for j in range(m)])
            for j in range(m):
                for i in range(self.k):
                    d.iloc[j, i] = self.distance(self.D.iloc[j], sample.loc[i])
                label = self.getMinIndex(d.ix[j])
                C[label].append(self.D.iloc[j])
            # find new mean vector
            newMeanVector = [pd.Series(index=self.D.columns.values) for i in range(self.k)]
            for i in range(self.k):
                newMeanVector[i] = self.getNewVectorValue(C[i])
                isNotEqual = False
                for a in self.D.columns.values[1:]:
                    if newMeanVector[i][a] != sample.ix[i][a]:
                        isNotEqual = True
                        break
                if isNotEqual:
                    isUpdate = True
                    # update mean vector
                    for a in self.D.columns.values[1:]:
                        sample.iloc[i][a] = newMeanVector[i][a]
            self.cluster = C


k = KMeans("data/4.0.csv", 3)
k.calculate()
k.cluster
