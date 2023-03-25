import pandas as pd
from flask import Flask, jsonify, request
from random import choice
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def get_predection1():
    print("Here1")
    new_prediction_data = request.form
    print(new_prediction_data.get("hsd"))
    print(new_prediction_data)

    df = pd.read_csv("data.csv")
    df.head()
    df.drop('Name', axis=1, inplace=True)
    df.head()
    df.State.nunique()
    df.State

    enc = OrdinalEncoder()
    State = enc.fit_transform(df.State.to_numpy().reshape(-1,1))

    df.State = State.reshape(1,-1)[0]
    df.State.unique()
    df.head()
    df.Sex.unique()
    df.info()

    answer = choice(['yes', 'no'])
    cols = ['Sex', 'Disability', 'Married', 'Scholarship', 'Transport Facilities', 'Toilet Facilities', 'Drinking Water Availability']

    from sklearn.preprocessing import OneHotEncoder
    oh_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_df = pd.DataFrame(oh_enc.fit_transform(df[cols]))

    df = pd.concat([df.drop(cols, axis=1), oh_df], axis=1)
    df.head()

    X = df.drop(['Dropout'], axis=1)
    y = df['Dropout']

    prediction = answer
    xgb = XGBClassifier()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, train_size=0.8)

    # xgb.fit(X_train, y_train)
    # y_pred = xgb.predict(X_valid)

    # print("accuracy score: ", accuracy_score(y_valid, y_pred))


    # Firestore
    return jsonify({'prediction': prediction, "uid" : new_prediction_data.get("uid")})


if __name__ == '__main__':
    app.run(debug=True)

print("Complete")