from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def classification_metrics(ypred,ytest,Xtrain,ytrain,lr_mn):
    print("Accuracy of training data:",lr_mn.score(Xtrain,ytrain))
    print("Test data results:")
    print("Accuracy:",accuracy_score(ytest, ypred))
    print(classification_report(ytest,ypred, digits=3))


def confusionmatrix(ypred,ytest):
    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True,fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()