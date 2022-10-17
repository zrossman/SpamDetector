from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

api = KaggleApi()
api.authenticate()
api.dataset_download_file('monizearabadgi/spambase', file_name = 'train_data.csv')

#Reading in our data set, and observing its shape
df = pd.read_csv('train_data.csv')
print(df.shape)
print()

#Taking a look at our columns
print(df.columns)
print()

#Lets get rid of the 'id' column
df = df.drop(['Id'], axis = 1)
print(df.columns)
print()

#Converting our features and our label into arrays
X = np.array(df.loc[:, 'word_freq_make': 'char_freq_#'])
y = np.array(df['ham'])

#Splitting our data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)

#Training our data
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Using our model to predict y_test using X_test
y_pred = classifier.predict(X_test)

#Evaluating our model
print('Accuracy:', classifier.score(X_test, y_test))
print()

print(y_test[:10])
print()
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Counting the true number of spam and ham in our test data
ham = 0
spam = 0
for i in range(len(y_test)):
    if y_test[i] == True:
        ham += 1
    else:
        spam += 1

print('Total Ham:', ham)
print('Total Spam:', spam)
print()

#Now we construct a method of taking an email, and making it suitable for our model
def formator(email):
    a_list = [' ', '3','0', '6', '5', '8', '7', '4', '1', '9', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a',
              's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x','c', 'v', 'b', 'n', 'm', 'Q', 'W', 'E', 'R', 'T', 'Y',
              'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ';',
              '(', '[','!', '$', '#']

    for character in email:
        if character not in a_list:
            email = email.replace(character, '')

    email = email.split(' ')
    total_words = len(email) - 1

    word_count = {}
    for word in email:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    relevant_words = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive',
                      'will', 'people', 'report', 'addresses', 'free', 'business' 'email', 'you', 'credit', 'your',
                      'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data',
                      '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project',
                      're', 'edu', 'table', 'conference', ';', '(', '[', '!', '$', '#', '*']

    final_format = []
    for element in relevant_words:
        if element in word_count:
            percentage = word_count[element] / total_words
            final_format.append(percentage)
        else:
            percentage = 0.0
            final_format.append(percentage)
    return final_format

#Creating a function that implements our model
def spam_or_ham(email):
    input = formator(email)
    input_array = np.array(input)
    input_array = np.reshape(input_array, (1, -1))
    result = classifier.predict(input_array)
    return result

#Testing(True indicates ham, false indicates spam
ham_email = '''Good morning! How are you? I am just writing to you today to see how things are going and if you
need anything. Let me know!'''
print(spam_or_ham(ham_email))

spam_email = '''Please give us credit card and address so we can see if  qualify for our 'money now'
program, where u receive money and rewards. Who doesn't like money money money! Thanks for business! Who doesn't like 
money money money! '''
print(spam_or_ham(spam_email))
