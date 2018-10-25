import json
import pandas as pd

def json2csv(jsonPath):
    with open(jsonPath, 'r') as f:
        label_list = json.load(f)
        print(type(label_list[0]))
        #for label in label_list:
        #    print(label)



def tmp():
    a = ['one','two','three']
    b = [1,2,3]
    english_column = pd.Series(a, name='english')
    number_column = pd.Series(b, name='number')
    predictions = pd.concat([english_column, number_column], axis=1)
    #another way to handle
    #save = pd.DataFrame({'english':a,'number':b})
    predictions.to_csv('b.txt',index=False,sep=',')
if __name__=='__main__':
    tmp()
    #json2csv(jsonPath = './dataset/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json')
