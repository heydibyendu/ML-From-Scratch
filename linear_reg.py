# Define the linear reg class
class LinearRegression():
    def __init__(self):
        # Initiate coef (slope and intercept) to 0
        self.slope = 0
        self.intercept = 0

    def fit(self, X, y):
        """Fit the model to the data"""
        # Calculate mean of X and y
        mean_X, mean_y = X.mean(), y.mean()

        
        





