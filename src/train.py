import numpy as np
from sklearn.model_selection import cross_val_score

def trainMod(models, X_train, y_train, X_test, y_test):
    for modelName, model in models.items():
        model.fit(X=X_train, y=y_train)
        scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
        print(f'{modelName} RMSE: ', np.mean(-scores))
    pass

"""
import sklearn.metrics as metrics
#WITHOUT CROSS VAL
    #y_pred = model.predict(X_test)
    #mse=metrics.mean_squared_error(y_test, y_pred) 
    #r2=metrics.r2_score(y_test, y_pred)
    #print(f'Model: {modelName}')
    #print('r2: ', round(r2,4))
    #print('RMSE: ', round(np.sqrt(mse),4))
"""