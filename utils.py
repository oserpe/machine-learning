from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def execute_grid_search_cv(model, X_train, X_test, y_train, y_test, params=dict(), cv=5):
    
    grid = GridSearchCV(model, params, refit = True, verbose = 3, n_jobs=-1, cv=5) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    print(grid.best_params_) 
    grid_predictions = grid.predict(X_test) 
    
    # print classification report 
    print(classification_report(y_test, grid_predictions)) 