import pandas as pd 
from src.clean import cleanDiamonds
from src.train import trainMod
from src.boosting import gridSearch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def main():

    diamonds_train = pd.read_csv("../license_datasets/diamonds_train.csv", delimiter=',')
    diamonds_test = pd.read_csv("../license_datasets/diamonds_test.csv", delimiter=',')
    
    #STEP 1 > Clean 
    diamonds_train = cleanDiamonds(diamonds_train, name='train')
    diamonds_test  = cleanDiamonds(diamonds_test, name='test')
    X = diamonds_train.drop(columns=['id','price']).values
    y = diamonds_train['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #STEP 2 > Train 
    models={'Linear Regression': LinearRegression(),
            'Decission Tree': DecisionTreeRegressor(),
            'Gradient Boosting':GradientBoostingRegressor(),
            'Random forest:': RandomForestRegressor()
    }
    trainMod(models, X_train, y_train, X_test, y_test)
    
    #STEP 3 > Boosting
    param_grid = {
    'max_depth': [10],
    'max_features': [2, 4, 5],
    'min_samples_leaf': [1,2,3],
    'random_state' : [12],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [2000]
    }
    gridSearch(model = RandomForestRegressor(), param_grid=param_grid, X=X, y=y, pred_data=diamonds_test)
  
if __name__ == '__main__':
    main()
