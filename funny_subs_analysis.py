
# coding: utf-8

# In[178]:

#Import and download the necessary toolkit

import operator
import numpy
import graphlab
from graphlab import SFrame

data = graphlab.SFrame.read_csv("50000Comments.csv", column_type_hints=[str, int, str, int])
data.dropna()


# In[179]:

data['target']=data['score']>10
funny_subs = ['funny']
#cut off first 50 words
data = data.filter_by(funny_subs,'subreddit')
data.head()
data.remove_columns(['subreddit', 'score', 'body_length'])


# In[180]:

body_array = data.select_column('body')
target_array = data.select_column('target')
print len(body_array), len(target_array)
array = []
for i in range(0, len(body_array)):
    tempArray = [body_array[i], target_array[i]]
    array.append(tempArray)
    i=i+1
print array[0:1]

import nltk
dataDict = {}
i=0
for i in range(len(array)):
    reader = array[i][0].decode('iso8859_15')
    tokens = nltk.word_tokenize(reader)
    freq_dist = nltk.FreqDist(tokens)
    dataDict[i]=freq_dist
#just for information
    i=i+1
    if i<5:
        print array[i][1], tokens[0:5]
        print freq_dist
#need to build a freq matrix


# In[181]:

# Add up all frequencies
features = {}
for i in range(len(array)):
  for feature in dataDict[i]:
    if feature not in features:
      features[feature] = 0
    features[feature]+=dataDict[i][feature]
#print features
# Select the most frequent ones
sorted_features = sorted(features.items(), key=operator.itemgetter(1), reverse=True)
sorted_features[130:200]


# In[182]:

# The numpy array of the most frequent 1000 terms
frequency_matrix = numpy.zeros((len(data),1000))
i=0
for i in range(len(array)):
    j=0
    for feature in sorted_features[0:1000]:
        frequency_matrix[i,j]=dataDict[i][feature[0]]
        j+=1
    i+=1
frequency_matrix[0:10,0:10]


# In[183]:

scoreArray = numpy.zeros((len(array), 1))
numOnes = 0
for i in range(len(array)):
    scoreArray[i] = target_array[i]
    if scoreArray[i]==1:
        numOnes+=1
print numOnes


# In[184]:

# append command
frequency_matrix = numpy.append(frequency_matrix,scoreArray,1)
numpy.savetxt("data1.csv",frequency_matrix)
# using GraphLab
import graphlab
data = graphlab.SFrame.read_csv("data1.csv",header=False,delimiter=" ",column_type_hints=int)
remove_list = []
for i in range(1,121):
    remove_word = 'X'+str(i)
    remove_list.append(remove_word)
#for i in ('X320', 'X380', 'X487', 'X648', 'X677', 'X685', 'X689', 'X709', 'X721', 'X722', 'X740', 'X742', 'X791', 'X811', 'X815', 'X827', 'X841', 'X844', 'X845', 'X854', 'X861', 'X863', 'X867', 'X868', 'X876', 'X882', 'X889', 'X891', 'X894', 'X897', 'X907', 'X908', 'X927', 'X932', 'X958', 'X961', 'X973', 'X986'):
#    remove_list.append(i)
data.remove_columns(remove_list)


# In[187]:

train, test = data.random_split(0.85, seed=10)
svm_model = graphlab.svm_classifier.create(train, target="X1001", penalty=1.2, feature_rescaling=False, lbfgs_memory_level=20, max_iterations=25, class_weights='auto')
predictions = svm_model.predict(test, output_type='margin')
predictions_numpy = numpy.array(predictions)
test_numpy = numpy.array(test["X1001"])
from sklearn import metrics
metrics.roc_auc_score(test_numpy,predictions_numpy)


# In[77]:

#train, test = data.random_split(0.85, seed=42)
#model=graphlab.boosted_trees_classifier.create(train, target="X1001", max_iterations=10, max_depth=5, step_size=0.8)
#predictions = model.predict(test)
#graphlab.evaluation.confusion_matrix(test["X1001"], predictions)


# In[ ]:



