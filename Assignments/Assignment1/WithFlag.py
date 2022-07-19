class WeightedKNNClassifier:
    """
    Class to store data for regression problems 
    """
    def __init__(self, k):
        """
        Creates a kNN instance
        :param k: The number of nearest points to consider in classification
        """
        
        # Import and build the BallTree on training features 
        self._k = k
        self._model = None
        self._y = None
        
        self._distanceFlag=None
        # Boolean flag indicating whether to do distance weighting (if you want to make it versatile)
        
    def fit(self, features, labels):
        
        # BEGIN Workspace 4.1
        self._model = sklearn.neighbors.BallTree(features)
        self._y = np.array(labels)

        #END Workspace 4.1

        # Should be used to map set of classes to {0,1,..C-1} where C is the number of classes
        classes = list(set(list(labels)))
        self.label_to_index = dict(zip(sorted(classes), range(len(classes))))
        self.index_to_label = dict(zip(range(len(classes)), sorted(classes)))
        
        self.distanceFlag=True
        return self



    def majority_vote(self, neighbor_indices, neighbor_distances=None):
        """
        Given indices of nearest neighbors in training set, return the majority label. 
        Break ties by considering 1 fewer neighbor until a clear winner is found. 

        :param neighbor_indices: The indices of the K nearest neighbors in self.X_train 
        :param neighbor_distances: Corresponding distances from query point to K nearest neighbors. 
        """
        
        
        # YOUR CODE HERE
        #BEGIN Workspace 4.2
        labels = [] #REPLACE
        #END Workspace 4.2
        if self.distanceFlag==False:
            #mostCommonLabel=max(self.counts.items(), key=operator.itemgetter(1))[0]
            myList=[]
            for i in neighbor_indices:
                myList.append(self._y[i])
            for i in myList:
                #print("i=",i)
                tempCount=Counter(i)
                length_i=len(i)
                temp_Label=None
                tempMax=max(tempCount.values())
                tempList=[]
                tempMaxKey=[]
                for j in tempCount:
                    if tempCount[j]==tempMax:
                        tempMaxKey.append(j)
                while temp_Label==None and length_i>0:
                    #print("tempCount=",tempCount)
                    for j in tempCount:
                        if tempCount[j]==tempMax:
                            tempList.append(j)
                    if len(tempList)>1:
                        length_i=length_i-1
                        continue
                    temp_Label=tempMaxKey[0]
                if length_i<1:
                    temp_Label=i[0]
                labels.append(temp_Label)
            return labels
        elif self.distanceFlag==True:
            myList=[]
            for i in neighbor_indices:
                myDict={}
                myList.append(self._y[i])
                for index,j in enumerate(i):
                    if neighbor_distances[index]==0:
                        labels.append(j)
                        break
                    temp_weight=1/neighbor_distances[index]
                    if j not in myDict:
                        myDict[j]=temp_weight
                    else:
                        myDict[j]+=temp_weight
                labels.append(max(myDict.items(), key=operator.itemgetter(1))[0])
            #Cite from Qiuyang Wang        
            return labels
    
    def predict(self, features):
        """
        Given an np.array of query points, return y_hat, an np.array of predictions

        :param features: an (m x p) dimension np.array of points to predict labels for
        """
        labels =[]
        # YOUR CODE HERE
        #BEGIN Workspace 4.3
        #TODO: predict labels
        #END Workspace 4.3
        for i in features:
            distance,index=self._model.query(i.reshape(1, -1), k = self._k)
            labels.append(self.majority_vote(index))
        labels=np.array(labels)    
        return labels
    
    
    def confusion_matrix(self, features_test, labels_test):
        """
        Generate a confusion matrix for the given test set
        PARAMETERS
        testX - an np.array of feature vectors of test points
        testY - the corresponding correct classifications of our test set
        RETURN
        C - an N*N np.array of counts, where N is the number of classes in our classifier
        """
        c_matrix = np.zeros((len(self.counts),(len(self.counts))))
        #BEGIN Workspace 4.4.a
        #TODO: Run classification for the test set, compare to test answers, and add counts to matrix
            
        #END Workspace 4.4.a
        for i in range(len(features_test)):
            #print("features_test=",features_test[i])
            #print("self.predict(features_test[i])[i]=",self.predict(features_test[[i]]))
            tempPred=self.predict(features_test[[i]])
            c_matrix[tempPred,labels_test[i]]+=1
        return c_matrix
    
    def accuracy(self, features_test, labels_test):
        """
        Generate an accuracy score for the classifier based on the confusion matrix
        PARAMETERS
        C - an np.array of counts
        RETURN
        score - an accuracy score
        """
        score = 0
        #BEGIN Workspace 4.4.b
        #TODO: Compute accuracy of the classification of features_test
        print("acc function")
        c_matrix = self.confusion_matrix(features_test, labels_test)
        score=np.sum(c_matrix.diagonal())/c_matrix.sum()
        #END Workspace 4.4.b
        return score 








X, y = sklearn.datasets.make_moons(n_samples=200, noise=0.19, random_state=42)
X_train, y_train = X[:100], y[:100]
X_test, y_test = X[100:], y[100:]


def show_decision_surface(model):
    """
    Helper function to visualize the decision surface of model
    :param model: Initialized KNNClassifier
    :return: None
    """
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    x_grid = np.arange(x_min, x_max, 0.1)
    y_grid = np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x_grid, y_grid)
    r1, r2 = xx.reshape(-1,1), yy.reshape(-1,1)
    grid = np.hstack((r1,r2))
    y_hat = model.predict(grid)
    zz = y_hat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='PiYG')
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

kk = 8
model = WeightedKNNClassifier(k=kk)
model = model.fit(X_train, y_train)
plt.title("Training decision surface for k = {}".format(kk))
show_decision_surface(model)







myModel=NumbersWeighted(k=5)
X_train = np.array([[0,3],[2,3],[3,3],[-1,2],[1,2],[3,2],[0,1],[.5,1], [.5,1], [2,1], [-1,-1], [0,-1], [3,-1]])
y_train = np.array([1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1])








#BEGIN Workspace 4.5
#TODO: Run multiple comparisons
repetitions = 10
ks = [1, 2, 4, 5, 6, 32, 50] # Add more
results_simple = np.zeros((len(ks),repetitions))
results_weighted = np.zeros((len(ks),repetitions))

for i, k in enumerate(ks):
    print(k,end =": ")
    for j in range(repetitions):
        print(j, end=",")
        numbers = NumbersWeighted(k=k)
        numbers.distanceFlag=True
        results_weighted[i,j] = numbers.accuracy() # TODO get accuracy
        numbers.classifier = KNNClassifier(k=numbers._k)
        results_simple[i,j] = numbers.accuracy() #TODO get accuracy
    print("")

#TODO Average results over multiple runs and report best classifier
    
#END Workspace 4.5
#print(results_weighted)
#print(results_simple)

# print(results_weighted.mean())
# print(results_simple.mean())


weightMean=[]
simpleMean=[]
for i in results_weighted:
    weightMean.append(i.mean())
    
for i in results_simple:
    simpleMean.append(i.mean())

plt.plot(ks,weightMean,label="weighted KNN")
#plt.plot(ks,simpleMean,label="Regular KNN")
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.title('Relationship between k value and accuracy')
plt.legend()
plt.show()









if self.distanceFlag==False:
            #mostCommonLabel=max(self.counts.items(), key=operator.itemgetter(1))[0]
            myList=[]
            for i in neighbor_indices:
                myList.append(self._y[i])
            for i in myList:
                #print("i=",i)
                tempCount=Counter(i)
                length_i=len(i)
                temp_Label=None
                tempMax=max(tempCount.values())
                tempList=[]
                tempMaxKey=[]
                for j in tempCount:
                    if tempCount[j]==tempMax:
                        tempMaxKey.append(j)
                while temp_Label==None and length_i>0:
                    #print("tempCount=",tempCount)
                    for j in tempCount:
                        if tempCount[j]==tempMax:
                            tempList.append(j)
                    if len(tempList)>1:
                        length_i=length_i-1
                        continue
                    temp_Label=tempMaxKey[0]
                if length_i<1:
                    temp_Label=i[0]
                labels.append(temp_Label)
            return labels
        else:
#             print("neighbor_distances",neighbor_distances)
#             print(print("neighbor_distances[0][0]",neighbor_distances[0][0]))
            #print("self._y",self._y)
            myList=[]
            
            #print(neighbor_distances)
            for i in neighbor_indices:
                myList.append(self._y[i])
            
            myDict={}
            for index, i in enumerate(neighbor_indices[0]):
#                 print("index=",index)
#                 print("i=",i)
                if neighbor_indices[0][index]==0:
                    labels.append(self._y[0])
                    break
                temp_weight=1/neighbor_distances[0][index]
                #print("hereeeee")
                if self._y[i] not in myDict:
                    myDict[self._y[i]]=temp_weight
                else:
                    myDict[self._y[i]]+=temp_weight
                labels.append(max(myDict.items(), key=operator.itemgetter(1))[0])
