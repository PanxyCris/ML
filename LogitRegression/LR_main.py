import numpy as np


class Logit:
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def loadDataSet(self, dataMatName, labelName):
        dataMat = []
        labelMat = []

        dr = open(dataMatName)
        for line in dr.readlines():
            lineArr = line.strip().split()
            lineList = [];
            lineList.append(1)  # beta
            for col in lineArr:
                lineList.append(float(col))
            dataMat.append(lineList)

        lr = open(labelName)
        for label in lr.readlines():
            labelMat.append(int(label))

        return dataMat, labelMat


    def train(self, dataSet, labels):
         dataMat = np.mat(dataSet)  # turn dataSet to the matrix
         labelMat = np.mat(labels).transpose()  # turn labels to the matrix and T
         m, n = np.shape(dataSet)  # row and column
         alpha = 0.01
         maxIter = 500
         weights = np.ones((n, 1))
         for i in range(maxIter):  # Iteration
             h = self.sigmoid(dataMat * weights)
             error = h - labelMat
             weights = weights - alpha * dataMat.transpose() * error
         return weights

    def loadStr(self, type, value):
        return 'assign2_dataset/page_blocks_' + type + '_' + value + '.txt'

    def test(self):
        trainDataSet, trainLabelSet = self.loadDataSet(self.loadStr('train','feature'), self.loadStr('train','label'))
        weights = self.train(trainDataSet, trainLabelSet)
        testDataSet, testLabelSet = self.loadDataSet(self.loadStr('test','feature'), self.loadStr('test','label'))
        dataMat = np.mat(testDataSet)
        labelMat = np.mat(testLabelSet).transpose()
        y = self.sigmoid(dataMat * weights)
        error = np.abs(y - labelMat)
        count = 0
        for data in error:
            if data == 0:
                count += 1
        return count/len(error)



logit = Logit()
print(logit.test())

