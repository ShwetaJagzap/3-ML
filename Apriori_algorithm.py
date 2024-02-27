# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:15:15 2023

@author: user
"""

#pip install mixtend
from mlxtend.frequent_patterns import apriori,association_rules
#here we are going to use transactional data wherein size of each
#row is not
#we can not use pandas to load unstructured data here function called open()
#create an empty list

groceries=[]
with open("groceries.csv") as f:groceries=f.read()
#splitting the data inyo seperate transaction using seperator it is comma
#we can use new line character "\n"
groceries=groceries.split("\n")
#Earlier groceries datastructure was in string format,
#now it will change to 
#9836,each item is comma separated
#our main aim is to calculate #A,#C
#we will have to seperate out each item from transaction
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
#split function will separate each item from each list whereever it will find
#in order to generate association rules,you can directly it will find in order
#now let us sepearte out each from the groceries list
all_groceries_list=[i for item in groceries_list for i in item]

#you will get all the items occured in all transactions
#we will get 43368 items in various transcations

#now let us count the frequency of each item
#we will import collections package which has counter function which will

from collection import Counter
item_frequencies=Counter(all_groceries_list)
#item frequencies is basically dictionary having x[0] as key and
#x[1]=values
#we want to access values and sort based on the count that occured in it
#now let us sort these frequencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
#when we excute this,item frequencies will be in sorted form,in the form tuple
#item name with count
#Let us seperate out items and their count
items=list(reversed([i[0] for i in item_frequencies]))
#this is list comprehension for each item in item frequencies access the key
frequencies=list(reversed([i[1] for i in item_frequencies]))
#here you will get count of purchase of each item

#now let us plot bar graph of item frequencies 
import matplotlib.pyplot as plt
#here we are taking frequencoes from zero to 11,you can try 0-15 or any other
plt.bar(height=frequencies[0:11],x=list(range(0,11)))
plt.xtricks(list(range(0,11),items[0:11]))
#plt.xtricks,you can specify a rotation for the tick
#labels in degrees or with keywords
plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#Now let us try to establish association rule mining
#we have groceries list in the list format we need to convert it into dataframe
groceries_series=pd.DataFrame(pd.Series(groceries_list))
#now we will get dataframe of size 9836X1 size ,column
#comprises of multiple items
#we had extra row created check the groceries_series
#last row is empty let us first delete it
groceries_series=groceries_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and column 0 to all
#groceries series has column having name 0,let us rename as transaction
groceries_series.columns=["Transcations"]
#Now we will have to apply 1-hot encoding,before that in
# ',' let us seperate it will '*'
x=groceries_series['Transactions'].str.join(sep='*')
#check the x in variable explorer which has * seperator rather tahn ','
x=x.str.get_dummies(sep='*')
#you will get one hot encoded dataframe of size 9835X169
#this is our input data to apply to apriori algorithm
#it will generate 169 rules ,min support values
#is  0.0075(it must be between 0 tol)
#you can give any number but must be between 0 and 1
frequent_itemsets=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
#you will get support values for 1,2,3 and 4 items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in decending order
#Even EDA was also have the same trend in EDA there was count
#and and here it is support values
#we will generte association rules,this association
#of each and every combination
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
#this generate association rules of size 1189X9 columns
#comprizes of antesents ,consequents
rules.heads(20)
rules.sort_values('lift',ascending=False).head(10)