from telegram.update import Update
from telegram.ext.updater import Updater
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv(r"C:\Users\gopik\Downloads\Training.csv").dropna(axis = 1)
test_data = pd.read_csv(r"C:\Users\gopik\Downloads\Testing.csv").dropna(axis=1)

updater = Updater('5633741085:AAEuAbkhJ3d0UsvbVbdJlphxx4sCH5ji3n0', use_context=True)
# dispatcher=updater.dispatcher

user_api = 'd91a74f331bbb8b424b920304262c616'

# def desease_predict(update: Update, context: CallbackContext):
#     update.message.reply_text("hi")
    # data = pd.read_csv(r"C:\Users\gopik\Downloads\Training.csv").dropna(axis = 1)
disease_counts = data["prognosis"].value_counts()
# print(disease_counts)
temp_df = pd.DataFrame({
"Disease": disease_counts.index,
"Counts": disease_counts.values
})
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
models = {
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10,n_jobs = -1,scoring = cv_scoring)
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

cf_matrix = confusion_matrix(y_test, preds)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)

cf_matrix = confusion_matrix(y_test, preds)

#Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
cf_matrix = confusion_matrix(y_test, preds)
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)
# test_data = pd.read_csv(r"C:\Users\gopik\Downloads\Testing.csv").dropna(axis=1)
test_data.head()
test_X = test_data.iloc[:, :-1]
test_Y = encoder.fit_transform(test_data["prognosis"])

svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [mode([i,j,k])[0][0] for i,j,k in zip(svm_preds, nb_preds, rf_preds)]

symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
data_dict = {
   "symptom_index":symptom_index,
   "predictions_classes":encoder.classes_
 }

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1,-1)
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = [rf_prediction,nb_prediction,nb_prediction,final_prediction]
    return predictions[3]

 # update.message.reply_text(predictDisease("Fluid Overload,Swelling Of Stomach,Swelled Lymph Nodes"))


# updater = Updater('5633741085:AAEuAbkhJ3d0UsvbVbdJlphxx4sCH5ji3n0', use_context=True)
# # dispatcher=updater.dispatcher
#
# user_api = 'd91a74f331bbb8b424b920304262c616'


def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello sir, Welcome to the Bot.Please write\
         /help to see the commands available.")



def do_something(user_input):
    if user_input in my_data.keys():
        v = my_data[user_input]
    b = " ".join(v)
    return b


def reply(update, context):
    user_input = update.message.text
    update.message.reply_text(predictDisease(user_input))



updater.dispatcher.add_handler(CommandHandler('start', start))
# updater.dispatcher.add_handler(CommandHandler('desease_predict', desease_predict))
updater.dispatcher.add_handler(MessageHandler(Filters.text, reply))
updater.start_polling()