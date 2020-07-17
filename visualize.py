import matplotlib.pyplot as plt
from classifiers import decision_tree,logistic_regression,kNN,neural_network

def extract_scores(scores):
    X,Y = [],[]

    for param,acc in scores:
        X.append(param)
        Y.append(acc)
    
    return X,Y


def main():
    # Decision Tree 
    dtree_scores = decision_tree(5)
    X,Y = extract_scores(dtree_scores)
    tree_max = max(Y)
    plt.plot(X,Y,'r')
    plt.axis([8, 22, 0.94, 0.98])
    plt.title('Decision Tree Classifier Accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.savefig('dtree.png')
    plt.clf()

    # Logistic Regression 
    log_scores = logistic_regression(5)
    X,Y = extract_scores(log_scores)
    log_max = max(Y)
    plt.plot(X,Y,'r')
    plt.axis([0, 1, 0.65, 0.8])
    plt.title('Logistic Regression Accuracy')
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.savefig('log.png')
    plt.clf()

    # kNN 
    k_scores = kNN(5)
    X,Y = extract_scores(k_scores)
    k_max = max(Y)
    plt.plot(X,Y,'r')
    plt.axis([0, 15, 0.95, 1.0])
    plt.title('kNN Accuracy')
    plt.xlabel('neighbors')
    plt.ylabel('accuracy')
    plt.savefig('knn.png')
    plt.clf()

    # neural network
    net_scores = neural_network(5)
    X,Y = extract_scores(net_scores)
    net_max = max(Y)
    plt.plot(X,Y,'r')
    plt.axis([0, 1.5, 0.95, 1.0])
    plt.title('Neural Network Accuracy')
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.savefig('net.png')
    plt.clf()

    # Final Results 
    best = [tree_max,k_max,net_max]
    models = ['DecisionTree','K-NN','NeuralNetwork']
    x_pos = [i for i, _ in enumerate(models)]

    plt.bar(x_pos, best, color='green')
    plt.ylim(0.95,1.0)
    plt.xlabel("Classifiers")
    plt.ylabel("Accuracy")
    plt.title("Best Accuracy from Models")


    plt.xticks(x_pos, models)

    plt.savefig('best.png')


if __name__ == '__main__':
    main()
