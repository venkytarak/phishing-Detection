

from api import get_prediction
from flask import Flask,render_template,request
app = Flask(__name__)
# model_path = r"E:/ml18internal/Malicious_URL_Prediction.h5"
model_path = r"E:/ml18internal/urlpred.pkl"

@app.route('/')
def hello_world():
   return render_template("index.html")
@app.route('/predict',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
     url=request.form['url']
     prediction = get_prediction(url,model_path)
     return render_template('index.html',url=prediction)
if __name__ == '__main__':
   app.run()
