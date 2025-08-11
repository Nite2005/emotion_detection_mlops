import unittest
import mlflow
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f'models:/{cls.model_name}/{cls.model_version}'
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        cls.holdout_data = pd.read_csv("data/processed/test_df.csv")
    
    @staticmethod
    def get_latest_modle_version(model_name):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=['staging'])
        return latest_version[0].version if latest_version else None
    
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    
    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_name_out())

        prediction = self.model.predict(input_df)
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_perfomance(self):
        #Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        #Predict using the new model
        y_pred_new = self.model.predict(X_holdout)

        #calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout,y_pred_new)
        precission_new = precision_score(y_holdout,y_pred_new)
        recall_new = recall_score(y_holdout,y_pred_new)
        f1_new = f1_score(y_holdout,y_pred_new)

        #define expected thresholds for the performance metrics
        expected_accuracy = 0.50
        expected_precision = 0.50
        expected_recall = 0.70
        expected_f1 = 0.70

        #Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new,expected_accuracy,f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precission_new,expected_precision,f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new,expected_recall,f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new,expected_f1,f'F1 score should be at least {expected_f1}')


if __name__ == "__main__":
    unittest.main()

