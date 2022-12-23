import flask
import os
import pickle
import pandas as pd
# from skimage import io
# from skimage import transform

app = flask.Flask(__name__, template_folder='dist')

# Init the path to get to our Random Forest Classifier Model
path_to_model = 'models/rfc.pkl'

# Load in our rfc model into the model variable
with open(path_to_model, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        age = flask.request.form['age_in']
        bmi = flask.request.form['bmi_in']
        glucose = flask.request.form['glucose_in']
        hypertension = flask.request.form['hypertension_in']
        heart_disease = flask.request.form['heart_in']

        # Convert the hypertension and heart disease inputs from a string into a 0 or 1
        if(hypertension == 'Yes'):
            hypertension = 1
        else:
            hypertension = 0
        
        if(heart_disease == 'Yes'):
            heart_disease = 1
        else:
            heart_disease = 0

        # Turn the inputs into a list
        list_of_inputs = [[age, bmi, glucose, hypertension, heart_disease]]
        # Using the list, feed it into the model so it can make a prediction
        pred = model.predict(list_of_inputs)
        # Grab the predicted probability for the prediction
        predicted_proba = model.predict_proba(list_of_inputs)
        predicted_proba = predicted_proba[0]

        # CCC
        # Change the output of pred to a string that will be printed out
        if(pred[0] == 1):
            probability = predicted_proba[1] * 100
            probability = round(probability, 2)
            prediction = "With a probability of {}% that you ARE AT RISK or a stroke".format(probability)
        else:
            prediction = "You are NOT at risk for a stroke"

        ## Return our inputs, list if inputs, and prediction message back to index.html
        return(flask.render_template('index.html', 
            returned_age=age,
            returned_bmi=bmi,
            returned_glucose=glucose,
            returned_hyp=hypertension,
            returned_heart=heart_disease,
            returned_list=list_of_inputs,
            returned_pred=prediction
            #returned_prob=predicted_proba
            ))

    # return(flask.render_template('index.html'))

if __name__ == '__main__':
    app.run(debug=True)
