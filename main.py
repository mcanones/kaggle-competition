import pandas as pd 
from src.clean import cleanDiamonds
from src.train import trainMod
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

def main():

    diamonds_train = pd.read_csv("../license_datasets/diamonds_train.csv", delimiter=',')
    diamonds_test = pd.read_csv("../license_datasets/diamonds_test.csv", delimiter=',')
    
    #STEP 1 > Clean 
    diamonds_train = cleanDiamonds(diamonds_train, name='train')
    diamonds_test  = cleanDiamonds(diamonds_test, name='test')

    X = diamonds_train.drop(columns=['id','price']).values
    y = diamonds_train['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #STEP 2 > Train General Models
    models={'Linear Regression': LinearRegression(),
            'Decission Tree': DecisionTreeRegressor(),
            'Random forest:': RandomForestRegressor(),
            'Gradient Boosting': LGBMRegressor(n_estimators=233, num_leaves=19, max_depth=8, min_child_samples=20)
            }
    trainMod(models, X_train, y_train, X_test, y_test)

    #STEP 2 > Grid Search
    param_grid = {
    'max_depth': [6,7,8],
    'num_leaves': list(range(40,100,1)),
    'learning_rate' : [0.05],
    'n_estimators': [1000]
    }
    grid_search =  RandomizedSearchCV(LGBMRegressor(), param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='neg_root_mean_squared_error', n_iter=32)
    grid_search.fit(X, y)
    print("RMSE: ", -grid_search.best_score_)
    print(grid_search.best_estimator_.get_params()) 
    
    #Submission
    id_ = diamonds_test['id']
    diamonds_test.drop(columns='id', inplace=True)
    y_pred = grid_search.predict(X=diamonds_test.values)
    final = pd.DataFrame({'id': id_, 'price': y_pred})
    print(final)
    final.to_csv('./output/final.csv', header = True, index=False) 
   
if __name__ == '__main__':
    main()
