from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle

df = pd.read_csv(r'breast_cancer\breast_cancer_data.csv')

dependent_variable=['diagnosis']
independent_variable=['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']

y=df['diagnosis']
x=df.drop(['diagnosis'],axis=1)



X_train, X_test, y_train, y_test = train_test_split( x,y , test_size = 0.2, random_state = 0)


classifier = RandomForestClassifier()
grid_values={'n_estimators':[50,80,100], 'max_depth': [3,5,7]}
classifier = GridSearchCV(classifier, param_grid = grid_values,scoring = 'roc_auc', cv=5)
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('breast_cancer\model.pkl', 'wb'))