import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("compare_two_whales.csv")

print(data)

plt.plot((data['First Whale'] + data['Second Whale']), data['Test Accuracy'], color = 'red' )
plt.xticks(rotation = 90) 
plt.xlabel('Whale combo') 
plt.ylabel('Test Accuracy') 
plt.show() 

