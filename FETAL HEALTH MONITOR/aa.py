from flask import Flask, request, render_template
import numpy as np
import pickle

model = pickle.load(open('fetal_health1.pkl', 'rb'))  # Corrected the file extension to 'pkl'
app = Flask(__name__)

@app.route('/')
def f():
    return render_template("index.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        # Collect form data and convert to float
        prolongued_decelerations = float(request.form['prolongued_decelerations'])
        abnormal_short_term_variability = float(request.form['abnormal_short_term_variability'])
        percentage_of_time_with_abnormal_long_term_variability = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
        histogram_variance = float(request.form['histogram_variance'])  # Corrected variable name
        histogram_median = float(request.form['histogram_median'])
        mean_value_of_long_term_variability = float(request.form['mean_value_of_long_term_variability'])
        histogram_mode = float(request.form['histogram_mode'])  # Corrected variable name
        accelerations = float(request.form['accelerations'])

        # Create a feature array
        X = np.array([[prolongued_decelerations, abnormal_short_term_variability,
                      percentage_of_time_with_abnormal_long_term_variability, histogram_variance,
                      histogram_median, mean_value_of_long_term_variability, histogram_mode, accelerations]])

        # Make a prediction using the model
        output = model.predict(X)
        output_label = ['Normal', 'Pathological', 'Suspect'][int(output[0])]

        return render_template('output.html', output=output_label)

if __name__ == '__main__':
    app.run(debug=True)
