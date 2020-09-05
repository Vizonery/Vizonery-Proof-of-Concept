import sys
import pickle
import pandas as pd


def return_result(param):
    loaded_model = pickle.load(open('model.sav', 'rb'))
    param = pd.DataFrame([param])
    result = loaded_model.predict_proba(param)
    return result[1]


if __name__ == '__main__':
    param1 = int(sys.argv[1])
    param2 = int(sys.argv[2])
    param3 = int(sys.argv[3])
    param4 = int(sys.argv[4])
    param5 = int(sys.argv[5])
    param = [param1,
             param2,
             param3,
             param4,
             param5]
    print(return_result(param))
