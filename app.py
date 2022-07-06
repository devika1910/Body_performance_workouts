from flask import Flask,request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
df=pd.read_csv("bodyPerformance.csv")
app = Flask(__name__)

#Deserialize
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html") #due to this function we are able to send our webpage to client(browser) - GET

@app.route('/predict',methods=['POST','GET'])  #gets inputs data from client(browser) to Flask Server - to give to ml model
def predict():
    features = [(x) for x in request.form.values()]
    #final = ['42', 'F', '164.5', '63.7', '32.2', '72', '135', '22.7', '0.8', '18', '146']

    print(features)
    #final = [np.array(features)] #['23','M','56']
    #final=final.tolist()
    final=[]
    for i in range(0,11):
        print(features[i])
        print(i)

        final.append(features[i])
        print(final[i])

    for i in range(0,11):
        if i==1:
            continue

        else:
            #final.replace(final[i],int(final[i]))

            final[i]=float(final[i])
            print(final[i])
    if final[1]=='M':
        #print(1)
        final[1]=1
    elif final[1]=='F':
        #print(0)
        final[1]=0


    '''if df.gender=='M':
        print(1)
    elif df.gender=='F':
        print(0)'''
    #our model was trained on Normalized(scaled) data
    df.replace("M", 0, inplace=True)
    df.replace("F", 1, inplace=True)
    X = df.iloc[:, :-1].values
    print(X[0])
    print(len(X[0]))

    sst=StandardScaler().fit(X)
    print(len(final))
    output = model.predict(sst.transform([final]))
    print(output)
    print(sst.transform([final]))

    if output[0]==1:
        return render_template('index.html',pred=f'Your are FIT and HEALTHY 100%')
    elif output[0]==2:
        return render_template('index.html',pred=f'your are FIT and HEALTHY')
    elif output[0]==3:
        return render_template('index.html',pred=f'You need to take care of your body')
    else:
        return render_template('index.html',pred=f'You need to concentrate more to maintain fit & Healthy Body')





if __name__ == '__main__':
    app.run(debug=True)