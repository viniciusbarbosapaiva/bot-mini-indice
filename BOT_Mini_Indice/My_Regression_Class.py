import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn import metrics
from yellowbrick.regressor import ResidualsPlot,PredictionError,CooksDistance
from yellowbrick.regressor.alphas import manual_alphas
import time

class MyRegressionClass:
    '''
    Class created by Tito with propose to train a regression machine learning process.
    '''
    def __init__(self,test_size=0.2, random_state=0):
        '''
        Parameters Initialization
        Parameters
    ----------
    stratify: array-like, default= y
    If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.
    
    random_state: int, RandomState instance or None, default=0
    Controls the shuffling applied to the data before applying the split. 
    Pass an int for reproducible output across multiple function calls.
    
    
        '''
        self.test_size = test_size
        self.random_state=random_state
        
    def TrainTestSet(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X, y , test_size=self.test_size, random_state=0)
        return X_train,X_test,y_train,y_test
    
    def LogisticRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Logistic Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        lr_regressor = LinearRegression(fit_intercept=True)
        lr_regressor.fit(X_train, y_train)
        y_pred = lr_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
                               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , LinearRegression
    
    def RidgeRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Ridge Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        rd_regressor = Ridge(alpha=50)
        rd_regressor.fit(X_train, y_train)
        y_pred = rd_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['Ridge Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , rd_regressor
    
    def LassoRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Lasso Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        la_regressor = Lasso(alpha=50)
        la_regressor.fit(X_train, y_train)
        y_pred = la_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['Lasso Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , la_regressor
    
    def PolynomialRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Polynomial Regressor '''
        k = X_test.shape[1]
        n = len(X_test)
        poly_reg = PolynomialFeatures(degree = 2)
        X_poly = poly_reg.fit_transform(X_train)
        lr_poly_regressor = LinearRegression(fit_intercept=True)
        lr_poly_regressor.fit(X_poly, y_train)
        y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(X_test))
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , lr_poly_regressor
    
    def SupportVectorRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Suport Vector Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        svr_regressor = SVR(kernel = 'rbf')
        svr_regressor.fit(X_train, y_train)
        y_pred = svr_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

        results = pd.DataFrame([['Support Vector RBF', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , svr_regressor
    
    def DecisionTreeRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Decision Tree Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        dt_regressor = DecisionTreeRegressor(random_state=0)
        dt_regressor.fit(X_train, y_train)
        y_pred = dt_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , dt_regressor
    
    def RandomForestRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Random Forest Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
        rf_regressor.fit(X_train, y_train)
        y_pred = rf_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , rf_regressor
    
    def AdaBoostingRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Ada Boosting Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        ad_regressor = AdaBoostRegressor()
        ad_regressor.fit(X_train, y_train)
        y_pred = ad_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , ad_regressor
    
    def GradientBoostingRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Gradient Boosting Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        gb_regressor = GradientBoostingRegressor()
        gb_regressor.fit(X_train, y_train)
        y_pred = gb_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , gb_regressor
    
    def XgBoostingRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Xg Boosting Regression'''
        k = X_test.shape[1]
        n = len(X_test)
        xgb_regressor = XGBRegressor()
        xgb_regressor.fit(X_train, y_train)
        y_pred = xgb_regressor.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100

        results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])
        return results, y_pred , xgb_regressor
    
    def AllEstimatorsTito(self,X_train,X_test,y_train,y_test):
        
        lr_regressor = LinearRegression(fit_intercept=True)
        rd_regressor = Ridge(alpha=50)
        la_regressor = Lasso(alpha=50)
        lr_poly_regressor= LinearRegression(fit_intercept=True)
        svr_regressor = SVR(kernel = 'rbf')
        dt_regressor = DecisionTreeRegressor(random_state=0)
        rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
        ad_regressor = AdaBoostRegressor()
        gb_regressor = GradientBoostingRegressor()
        xgb_regressor = XGBRegressor()

        model = [('Logistic Regression', lr_regressor),
                 ('Ridge Regression', rd_regressor),
                 ('Lasso Regression', la_regressor),
                 ('Polynomial Regression', lr_poly_regressor),
                 ('Support Vector RBF', svr_regressor),
                 ('Decision Tree Regression', dt_regressor),
                 ('Random Forest Regression', rf_regressor),
                 ('Ada Boosting', ad_regressor),
                 ('Gradient Boosting', gb_regressor),
                 ('Xg Boosting', xgb_regressor)]
        results = pd.DataFrame()
        y_predict = []
        
        for models,estimator in model:
            if models == 'Polynomial Regression':
                k = X_test.shape[1]
                n = len(X_test)
                poly_reg = PolynomialFeatures(degree = 2)
                X_poly = poly_reg.fit_transform(X_train)
                estimator.fit(X_poly, y_train)
                y_pred = estimator.predict(poly_reg.fit_transform(X_test))
                mae = metrics.mean_absolute_error(y_test, y_pred)
                mse = metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                r2 = metrics.r2_score(y_test, y_pred)
                adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
                model_results = pd.DataFrame([[models, mae, mse, rmse, r2,adj_r2]],
                                             columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score'])
                results = results.append(model_results, ignore_index = True)
                y_predict.append((models,list(y_pred)))
            
            else:
                k = X_test.shape[1]
                n = len(X_test)
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                mae = metrics.mean_absolute_error(y_test, y_pred)
                mse = metrics.mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                r2 = metrics.r2_score(y_test, y_pred)
                adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
                model_results = pd.DataFrame([[models, mae, mse, rmse, r2,adj_r2]],
                                             columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score'])
                results = results.append(model_results, ignore_index = True)
                y_predict.append((models,list(y_pred)))
        
        ##Ensemble Voting regressor
        voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),
                                                        ('rd', rd_regressor),
                                                        ('la', la_regressor),
                                                        ('lr_poly', lr_poly_regressor),
                                                        ('svr', svr_regressor),
                                                    ('dt', dt_regressor),
                                                    ('rf', rf_regressor),
                                                    ('ad', ad_regressor),
                                                    ('gr', gb_regressor),
                                                    ('xg', xgb_regressor)])

        for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,
                    rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):
            
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))
        
        #The Best Classifier
        the_best_estimator = results.sort_values(by='Adj. R2 Score',ascending=False).head(5)
        print('The best regressor is:')
        print('{}'.format(the_best_estimator))

        y_pred = pd.DataFrame(y_predict, columns=['Model', 'y_pred'])
        y_pred = y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
        y_pred = list(y_pred)
            
        model_df = pd.DataFrame(model, columns=['Model', 'Estimator'])
        estimator = model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
        estimator = list(estimator)
        
        return results, y_pred[0] , estimator[0]
    
    def ResidualPlotTito(self,model, X_train,X_test,y_train,y_test):
        '''
        Residuals, in the context of regression models, 
        are the difference between the observed value of the target variable (y) 
        and the predicted value (ŷ), i.e. the error of the prediction. 
        The residuals plot shows the difference between residuals on the 
        vertical axis and the dependent variable on the horizontal axis, 
        allowing you to detect regions within the target that may be susceptible to more or less error.
        '''
        visualizer = ResidualsPlot(model)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()                 # Finalize and render the figure
    
    def PredictionErrorTito(self,model, X_train,X_test,y_train,y_test):
        '''
        A prediction error plot shows the actual targets from the dataset against 
        the predicted values generated by our model. 
        This allows us to see how much variance is in the model. 
        Data scientists can diagnose regression models using this plot by comparing against the 45 degree line, 
        where the prediction exactly matches the model.
        '''
        visualizer = PredictionError(model)
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()                 # Finalize and render the figure
        
    def AlphaSelectionTito(self,model, X_train,y_train):
        '''
        Regularization is designed to penalize model complexity, 
        therefore the higher the alpha, the less complex the model, 
        decreasing the error due to variance (overfit). 
        Alphas that are too high on the other hand increase the error due to bias (underfit). 
        It is important, therefore to choose an optimal alpha such that the error is minimized in both directions.
        
        The AlphaSelection Visualizer demonstrates how different values of alpha influence model selection during the 
        regularization of linear models. Generally speaking, alpha increases the affect of regularization, 
        e.g. if alpha is zero there is no regularization and the higher the alpha, the more the regularization 
        parameter influences the final model.
        '''
        start = time.time()
        manual_alphas(model, X_train, y_train, cv=10)
        end = time.time()
        print("The creation of Alpha Selection graph has taken {}".format(time.strftime("%H:%M:%S",time.gmtime(end-start))))
        
    def CookDistanceTito(self,model, X_train,y_train):
        '''
        Cook’s Distance is a measure of an observation or instances’ influence on a linear regression. 
        Instances with a large influence may be outliers, 
        and datasets with a large number of highly influential points might not be suitable
        for linear regression without further processing such as outlier removal or imputation. 
        The CooksDistance visualizer shows a stem plot of all instances by index and their associated distance score, 
        along with a heuristic threshold to quickly show what percent of the dataset may be impacting OLS regression models.
        '''
        visualizer = CooksDistance()
        visualizer.fit(X_train, y_train)
        visualizer.show()





       