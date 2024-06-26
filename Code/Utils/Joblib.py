import joblib
import datetime as dt

class Joblib:
    
    def __init__(self) -> None:
        pass

    def save_model_to_joblib_folder(self, model : any, model_name : str):
        file_path = 'JoblibModels\\' + model_name +'_'+ str(dt.datetime.now().strftime("%d%m%Y%H%M%S")) + '.joblib'
        joblib.dump(model, file_path)
        print(f'Model saved successfully as {file_path}')
    
    def load_model_from_joblib_folder(self, file_name : str):
        file_path = r'JoblibModels/'+file_name+'.joblib'
        model = joblib.load(file_path)
        print(f'Model loaded successfully from {file_path}')
        return model
    
    def save_model_to_specified_path(self, model : any, path : str, model_name : str):

        file_path = r''+path+'\\' + model_name +'_'+ str(dt.datetime.now().strftime("%d%m%Y%H%M%S")) +'.joblib'
        joblib.dump(model , file_path)
        print(f'Model Saved Successfully as {file_path}')

    def load_model_from_specified_path(self, path : str, file_name : str):

        file_path = r''+path+'/'+file_name+'.joblib'
        model = joblib.load(file_path)
        print(f'Model loaded successfully from {file_path}')
        return model