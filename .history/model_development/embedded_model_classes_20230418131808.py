from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class DoubleLDA():
    def __init__(self):
        self.lda1 = LinearDiscriminantAnalysis()
        self.lda2 = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.lda1.fit(X, y)
        self.lda2.fit(X, y)

    def predict(self, X):
        pred1 = self.lda1.predict(X)
        pred2 = self.lda2.predict(X)
        
        for i in range(len(X)):
            if pred1[i] == pred2[i]:
                # if the predictions are the same, return that prediction
                prediction = pred1[i]
            else:
                # if the predictions are different, return the prediction of the first LDA object
                prediction = pred1[i]
            yield prediction


class DoubleLDA():
    def __init__(self):
        self.lda1 = LinearDiscriminantAnalysis()
        self.lda2 = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.lda1.fit(X, y)
        pred1 = self.lda1.predict(X)
        # Find the class with worse score
        score1 = self.lda1.score(X, y)
        classes = self.lda1.classes_
        score2 = score1
        worst_class = classes[0]
        for c in classes:
            if self.lda1.score(X[pred1 == c], y[pred1 == c]) < score2:
                score2 = self.lda1.score(X[pred1 == c], y[pred1 == c])
                worst_class = c
        self.lda2.fit(X[pred1 == worst_class], y[pred1 == worst_class])

    def predict(self, X):
        pred1 = self.lda1.predict(X)
        pred2 = self.lda2.predict(X)
        
        for i in range(len(X)):
            if pred1[i] == pred2[i]:
                # if the predictions are the same, return that prediction
                prediction = pred1[i]
            else:
                # if the predictions are different, return the prediction of the first LDA object
                prediction = pred1[i]
            yield prediction
