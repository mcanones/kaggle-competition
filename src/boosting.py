from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

def gridSearch(model, param_grid, X, y, pred_data):
    
    grid_search =  RandomizedSearchCV(model, param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='neg_root_mean_squared_error', n_iter=32)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_.get_params())

    #Submission
    id_ = pred_data['id']
    pred_data.drop(columns='id', inplace=True)
    y_pred = grid_search.predict(pred_data.values)
    final = pd.DataFrame({'id': id_, 'price': y_pred})
    final.to_csv('./output/final.csv', header = True, index=False)  
    pass