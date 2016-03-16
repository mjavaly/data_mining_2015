
# coding: utf-8

# In[112]:

#Import and download the necessary toolkit

import operator
import numpy
import graphlab
from graphlab import SFrame

data = graphlab.SFrame.read_csv("50000Comments11.csv", column_type_hints=[str, int, str, int])
data.dropna()


# In[113]:

data['target']=data['score']>10
gaming_subs = ['gaming','leagueoflegends','skyrim']
#cut off first 150 words
data = data.filter_by(gaming_subs,'subreddit')
data.head()
data.remove_columns(['subreddit', 'score', 'body_length'])


# In[114]:

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


# In[116]:

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
sorted_features[150:200]


# In[117]:

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


# In[118]:

scoreArray = numpy.zeros((len(array), 1))
numOnes = 0
for i in range(len(array)):
    scoreArray[i] = target_array[i]
    if scoreArray[i]==1:
        numOnes+=1
print numOnes


# In[119]:

# append command
frequency_matrix = numpy.append(frequency_matrix,scoreArray,1)
numpy.savetxt("data1.csv",frequency_matrix)
# using GraphLab
import graphlab
data = graphlab.SFrame.read_csv("data1.csv",header=False,delimiter=" ",column_type_hints=int)
remove_list = []
for i in range(1,151):
    remove_word = 'X'+str(i)
    remove_list.append(remove_word)
#for i in ('X751', 'X854', 'X880', 'X905', 'X963', 'X964', 'X988', 'X995'):
#    remove_list.append(i)
data.remove_columns(remove_list)


# In[127]:

train, test = data.random_split(0.95, seed=10)
svm_model = graphlab.svm_classifier.create(train, target="X1001", penalty=1.2, feature_rescaling=False, lbfgs_memory_level=20, max_iterations=25, class_weights='auto')
predictions = svm_model.predict(test, output_type='margin')
predictions_numpy = numpy.array(predictions)
test_numpy = numpy.array(test["X1001"])
from sklearn import metrics
metrics.roc_auc_score(test_numpy,predictions_numpy)


# In[11]:

#train, test = data.random_split(0.85, seed=42)
#model=graphlab.boosted_trees_classifier.create(train, target="X1001", max_iterations=10, max_depth=5, step_size=0.8)
#predictions = model.predict(test)
#graphlab.evaluation.confusion_matrix(test["X1001"], predictions)


# In[ ]:



