
# Preparing the text data

* Removal of stop words(such as "and","the","of"...)
* Lemmatization(such as "include","includes" and "included" -> "include")

# Creating word dictionary


```python
import glob
import os
from collections import Counter
import numpy as np
```


```python
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail)as m:
            for i,line in enumerate(m):
                if i == 2: # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    # Paste code for non-words removal here(code snippet is given below)
    remove_list = dictionary.keys()
    list_to_remove = list(remove_list)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary
```


```python
train_dir = "Machine Learning_Bayes_Lab/ling-spam/train-mails/"
```


```python
dictionary = make_Dictionary(train_dir)
```

# Feature extraction process


```python
def extract_features(mail_dir):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            docID = docID + 1
    return features_matrix
```


```python
msg_mail_dir = "Machine Learning_Bayes_Lab/msg"
spmsg_mail_dir = "Machine Learning_Bayes_Lab/spmsg"
```


```python
msg_features_matrix = extract_features(msg_mail_dir)
spmsg_features_matrix = extract_features(spmsg_mail_dir)
#print(spmsg_features_matrix.shape)
```

# Training the classifiers


```python
def conditionPro(features_matrix):
    raws,columns = features_matrix.shape
    condPro_list = []
    for col in range(columns):
        featureVal = 0
        for raw in range(raws):
            if features_matrix[raw][col] > 0:
                featureVal += 1
        condPro_list.append(featureVal/raws)
    return condPro_list
```


```python
msg_condPro_list = conditionPro(msg_features_matrix)
spmsg_condPro_list = conditionPro(spmsg_features_matrix)
```


```python
test_dir = "Machine Learning_Bayes_Lab/ling-spam/test-mails/"
```


```python
def test_mails_pro(test_dir):
    test_emails = [f for f in os.listdir(test_dir)]
    test_emails_sum = len(test_emails)
    test_pro_result = [] * 2
    test_msg_sum = 0
    test_spmsg_sum = 0
    for name in test_emails:
        names = name.split()
        if names[0].isalpha == False:
            test_msg_sum += 1
        else:
            test_spmsg_sum += 1
    test_pro_result.append(test_msg_sum/test_emails_sum)
    test_pro_result.append(test_spmsg_sum/test_emails_sum)
    return test_pro_result
```


```python
def extract_single_mail_features(mail):
        single_features_matrix = np.zeros((1,3000))
        with open(mail) as ma:
            for i, line in enumerate(ma):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                single_features_matrix[0, wordID] = words.count(word)
        return single_features_matrix
```

# Checking the performance

**Naive Bayes** : $$ P(S|W) = \frac {P(W|S)P(S)}{P(W)}$$ $$ P(S|W_1, W_2,...,W_n) = \frac {P(W_1, W_2,...,W_n|S)P(S)}{P(W_1, W_2,...,W_n)} \propto P(W_1, W_2,...,W_n|S)P(S)$$ 
**When all the features are independent **
$$ \propto P(W_1|S)P(W_2|S)...P(W_n|S)P(S) $$

* Compute $P(W_1|S),P(W_2|S),...,P(W_n|S)$ (use `conditionPro`)

    * Extract features from spam emails and ham emails, construct $351 * 3000$ features matrix
    * For every word in every test email(such as spam) $$ P(W_i|S) = \frac {the\ occurence\ \#\ of\ word\ in\ train\ spam\ emails}{the\ \#\ of\ train\ spam\ emails} $$
    * When $P(W_i|S) = 0$, we use `Laplacian smoothing` $$P(W_i|S) = \frac {n_1 + 1}{n + N}$$ 
    
    $n_1$ represent the occurence $\#$ of word(in test emails) in train emails. 
    
    $n$ represent the occurance $\#$ of all words in train emails
    
    $N$ represent the $\#$ of words in train emails.


```python
test_emails = [f for f in os.listdir(test_dir)]
prior = test_mails_pro(test_dir)
pre_right = 0
fn = 0
tp = 0
fp = 0
for mail in test_emails:
    msg_likelyhood = 1
    spmsg_likelyhood = 1
    names = mail.split('.')
    if 'spmsgc' in names[0]:
        test_result = 1
    else:
        test_result = 0
    mail = os.path.join(test_dir, mail)
    mail_features = extract_single_mail_features(mail)
    mail_features_index = [i for i in range(3000) if mail_features[0][i]>0]
    msg_features_pro = [msg_condPro_list[j] for j in mail_features_index]
    spmsg_features_pro = [spmsg_condPro_list[j] for j in mail_features_index]
    # Laplacian smoothing
    for i in range(len(mail_features_index)):
        if msg_features_pro[i] == 0:
            words_times = 0
            words_all_times = 0
            for j in range(351):
                words_times += msg_features_matrix[j, mail_features_index[i]]
                for k in range(3000):
                    words_all_times += msg_features_matrix[j, k]
            msg_features_pro[i] = (words_times+1)/(3000 + words_all_times)
        msg_likelyhood *= msg_features_pro[i] 
        if spmsg_features_pro[i] == 0:
            words_times = 0
            words_all_times = 0
            for j in range(351):
                words_times += spmsg_features_matrix[j, mail_features_index[i]]
                for k in range(3000):
                    words_all_times += spmsg_features_matrix[j,k]
            spmsg_features_pro[i] = (words_times+1)/(3000 + words_all_times)
        spmsg_likelyhood *= spmsg_features_pro[i]
    msg_posterior = msg_likelyhood * 0.5
    spmsg_posterior = spmsg_likelyhood * 0.5
    if msg_posterior < spmsg_posterior:
        pre_result = 1
    else:
        pre_result = 0
    if pre_result == test_result:
        pre_right += 1
    if pre_result == 1 and test_result == 0:
        fn += 1
    if pre_result == 0 and test_result == 0:
        tp += 1
    if pre_result == 0 and test_result == 1:
        fp += 1
accuracy = pre_right/len(test_emails)
# 召回率：所有正例中正确的概率
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1 = 2 * precision * recall / (precision + recall)
print('accuracy',accuracy,'recall',recall,'f1',f1)
```

    accuracy 0.9692307692307692 recall 0.9846153846153847 f1 0.9696969696969696


# Performance

|performance|value|
|:---------:|:---:|
|accurancy|0.9692|
|recall|0.9846|
|f1|0.9696|

The accurancy is about 84.23% when we do not use `Laplacian smoothing`. Obviously, accuracy depends on conditional probability, when the conditional probility is 0, it prones on ham email. 

Besides, naive bayes as classifier depends on the independence of features. So the results are not ideal and realistic. What's more, there are some issues need to be considered, such as the new words, number of multiples(conditional probability)。
