import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def data_info(data):
    data.info()
    print(data)

def classify_features():
    col = list(data.columns)
    for i in col:
        if len(data[i].unique()) > 6:
            numerical_features.append(i)
        else:
            categorical_features.append(i)

    print('Categorical Features :', *categorical_features)
    print('Numerical Features :', *numerical_features)

def transform_features():
    le = LabelEncoder()

    for features in categorical_features:

        df1[features] = le.fit_transform(df1[features])


    mms = MinMaxScaler()  # Normalization
    ss = StandardScaler()  # Standardization

    # for features in numerical_features:
    #     df1[features] = ss.fit_transform(df1[[features]])

    df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])
    df1['Age'] = ss.fit_transform(df1[['Age']])
    df1['RestingBP'] = ss.fit_transform(df1[['RestingBP']])
    df1['Cholesterol'] = ss.fit_transform(df1[['Cholesterol']])
    df1['MaxHR'] = ss.fit_transform(df1[['MaxHR']])

    print(df1.head())

def make_featureMD():
    feature_MD = pd.DataFrame({
        'kaggle_ID': kaggle_id,
        'feature_name': feature_name,
        'categorical_boolean': categorical_boolean,
        'numerical_boolean': numerical_boolean,
        'chi_squared_score': chi_square_score,
        'ANOVA_score': ANOVA_scor,
        'dropped_boolean': dropped_boolean
    })

    # convert file metadata to csv
    featureNAME = "Dataset_examples/MachineLearningMD/feature_MLMD"
    feature_MD.to_csv(featureNAME, index=False, header=True)

def make_paramMD():
    param_MLMD = pd.DataFrame({
        'kaggle_ID': kaggle_id,
        'ML_name': MLparam_name,
        'param_name': param_name,
        'param_value': param_value,
    })

    paramNAME = "Dataset_examples/MachineLearningMD/param_MLMD"
    param_MLMD.to_csv(paramNAME, index=False, header=True)


def make_modelMD():
    ML_results_model_MD = pd.DataFrame({
        'kaggle_ID': did,
        'ML_name': ML_name,
        'accuracy': accuracy,
        'cross_validation_score': cross_validation_score,
        'ROC_AUC_score': ROC_AUC_score
    })

    modelNAME = "Dataset_examples/MachineLearningMD/model_results_MLMD"
    ML_results_model_MD.to_csv(modelNAME, index=False, header=True)

def make_model_evaluationMD():
    ML_results_model_evaluation_MD = pd.DataFrame({
        'Kaggle_ID': model_eval_kaggleID,
        'ML_name': model_eval_ML_name,
        'variable': model_eval_name,
        'precision': model_eval_precision,
        'recall': model_eval_recall,
        'f1-score': model_eval_f1_score,
        'support': model_eval_support
    })

    model_evalNAME = "Dataset_examples/MachineLearningMD/model_eval_results_MLMD"
    ML_results_model_evaluation_MD.to_csv(model_evalNAME, index=False, header=True)


def filter_params(df1):
    numerical_df = df1.loc[:, numerical_features]
    categorical_df = df1.loc[:, categorical_features]
    categorical_df = categorical_df.drop(target_param, axis=1)
    corr_numerical = numerical_df.corrwith(df1[target_param]).to_frame()
    corr_category = categorical_df.corrwith(df1[target_param]).to_frame()
    for index, row in corr_category.iterrows():
        print(index)
        if abs(row[0]) < 0.2:
            df1 = df1.drop(index, axis=1)
            dropped_boolean.append(True)
        else:
            dropped_boolean.append(False)
    for index, row in corr_numerical.iterrows():
        print(index)
        if abs(row[0]) < 0.2:
            df1 = df1.drop(index, axis=1)
            dropped_boolean.append(True)
        else:
            dropped_boolean.append(False)
    return df1


def mean_value_features(data):
    yes = data[data['HeartDisease'] == 1].describe().T
    no = data[data['HeartDisease'] == 0].describe().T
    colors = ['#F93822', '#FDD20E']

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(yes[['mean']], annot=True, cmap=colors, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f', )
    plt.title('Heart Disease');

    plt.subplot(1, 2, 2)
    sns.heatmap(no[['mean']], annot=True, cmap=colors, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
    plt.title('No Heart Disease');

    fig.tight_layout(pad=2)
    plt.show()





def model(classifier, MLA_name):
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    ML_name.append(MLA_name)
    did.append('12345')
    accuracy.append(accuracy_score(y_test, prediction))
    print("Accuracy : ", '{0:.2%}'.format(accuracy_score(y_test, prediction)))
    cross_validation_score.append(cross_val_score(classifier,x_train,y_train, cv=cv, scoring = 'roc_auc').mean())
    print("Cross Validation Score : ",
          '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc').mean()))
    ROC_AUC_score.append(roc_auc_score(y_test, prediction))
    print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc_score(y_test, prediction)))
    plot_roc_curve(classifier, x_test, y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()


def model_evaluation(classifier, ML_name):

    # Classification Report
    class_report = classification_report(y_test, classifier.predict(x_test))
    class_report_dict = classification_report(y_test, classifier.predict(x_test), output_dict=True)
    for index in class_report_dict:
        if index == 'accuracy':
            model_eval_kaggleID.append('12345')
            model_eval_ML_name.append(ML_name)
            model_eval_name.append(index)
            model_eval_precision.append(class_report_dict[index])
            model_eval_recall.append('NULL')
            model_eval_f1_score.append('NULL')
            model_eval_support.append('NULL')
        else:
            model_eval_kaggleID.append('12345')
            model_eval_ML_name.append(ML_name)
            model_eval_name.append(index)
            model_eval_precision.append(class_report_dict[index]['precision'])
            model_eval_recall.append(class_report_dict[index]['recall'])
            model_eval_f1_score.append(class_report_dict[index]['f1-score'])
            model_eval_support.append(class_report_dict[index]['support'])

    print(class_report_dict)

def lr_exec_and_MD(random_state, C, penalty):
    classifier_lr = LogisticRegression(random_state=random_state, C=C, penalty=penalty)
    params = classifier_lr.get_params()
    MLA_name = 'LogisticRegression'
    param_extraction(params, MLA_name)

    model(classifier_lr,MLA_name)
    model_evaluation(classifier_lr, MLA_name)

def svc_exec_and_MD(kernel,C):
    classifier_svc = SVC(kernel=kernel, C=C)
    params = classifier_svc.get_params()
    MLA_name = 'SVC'
    param_extraction(params, MLA_name)

    model(classifier_svc, MLA_name)
    model_evaluation(classifier_svc, MLA_name)

def dt_exec_and_MD(random_state, max_depth, min_samples_leaf):
    classifier_dt = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    params = classifier_dt.get_params()
    MLA_name = 'Decision Tree Classifier'
    param_extraction(params, MLA_name)

    model(classifier_dt, MLA_name)
    model_evaluation(classifier_dt,MLA_name)

def rf_exec_and_MD(max_depth, random_state):
    classifier_rf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    params = classifier_rf.get_params()
    MLA_name = 'Random Forest Classifier'
    param_extraction(params, MLA_name)

    model(classifier_rf, MLA_name)
    model_evaluation(classifier_rf, MLA_name)

def knn_exec_and_MD(leaf_size, n_neighbors, p):
    classifier_knn = KNeighborsClassifier(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    params = classifier_knn.get_params()
    MLA_name = 'K Nearest Neighbour'
    param_extraction(params, MLA_name)

    model(classifier_knn, MLA_name)
    model_evaluation(classifier_knn, MLA_name)

def chi_squared_score(df1):
    features = df1.loc[:, categorical_features[:-1]]
    target = df1.loc[:, categorical_features[-1]]

    best_features = SelectKBest(score_func=chi2, k='all')
    fit = best_features.fit(features, target)

    featureScores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['Chi Squared Score'])


    first = featureScores.iloc[0]
    print('line 73')
    print(featureScores)
    for index, row in featureScores.iterrows():
        kaggle_id.append(12345)
        feature_name.append(index)
        categorical_boolean.append('True')
        numerical_boolean.append('False')
        chi_square_score.append(str(row['Chi Squared Score']))
        ANOVA_scor.append('NULL')
        #print(featureScores.at[index, row['Chi Squared Score']])
        print(str(index) + ":: " + str(row['Chi Squared Score']) )

def ANOVA_score(df1):
    features = df1.loc[:, numerical_features]
    target = df1.loc[:, categorical_features[-1]]

    best_features = SelectKBest(score_func=f_classif, k='all')
    fit = best_features.fit(features, target)

    featureScores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['ANOVA Score'])
    print(featureScores)
    for index, row in featureScores.iterrows():
        kaggle_id.append(12345)
        feature_name.append(index)
        categorical_boolean.append('False')
        numerical_boolean.append('True')
        chi_square_score.append('NULL')
        ANOVA_scor.append(str(row['ANOVA Score']))
        print(str(index) + ":: " + str(row['ANOVA Score']))

def param_extraction(params, model_name):
    for parameter_name, parameter_value in params.items():
        kaggle_id.append('12345')
        MLparam_name.append(model_name)
        param_name.append(parameter_name)
        param_value.append(parameter_value)

# def callGraph():
#     from pycallgraph import PyCallGraph
#     from pycallgraph.output import GraphvizOutput
#
#     graphviz = GraphvizOutput()
#     graphviz.output_file = 'basic1.png'
#     with PyCallGraph(output=graphviz)


if __name__ == "__main__":
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    graph = GraphvizOutput()
    graph.output_file = "file3.png"

    with PyCallGraph(output = graph):



        data = pd.read_csv('Input_data/heart_failure_prediction/heart.csv')

        target_param = "HeartDisease"
        categorical_features = []
        numerical_features = []
        classify_features()

        df1 = data.copy(deep=True)
        transform_features()



        #feature_MD
        kaggle_id = []
        feature_name = []
        categorical_boolean = []
        numerical_boolean = []
        chi_square_score = []
        ANOVA_scor = []
        dropped_boolean = []

        chi_squared_score(df1)
        ANOVA_score(df1)

        df1 = filter_params(df1)

        make_featureMD()


        features = df1[df1.columns.drop([target_param])].values
        target = df1[target_param].values
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=2)

        #param_MLMD
        kaggle_id = []
        MLparam_name = []
        param_name = []
        param_value = []

        #model_results_MLMD
        did = []
        ML_name = []
        ML_model = []
        accuracy = []
        cross_validation_score = []
        ROC_AUC_score = []

        #model_evaluation_results_MLMD
        model_eval_kaggleID = []
        model_eval_ML_name = []
        model_eval_name = []
        model_eval_precision = []
        model_eval_recall = []
        model_eval_f1_score = []
        model_eval_support = []



        lr_exec_and_MD(random_state=0, C=10, penalty='l2')
        svc_exec_and_MD(kernel='linear', C=0.1)
        dt_exec_and_MD(random_state=1000, max_depth=4, min_samples_leaf=1)
        rf_exec_and_MD(max_depth=4, random_state=0)
        knn_exec_and_MD(leaf_size=1, n_neighbors=3, p=1)


        make_paramMD()
        make_modelMD()
        make_model_evaluationMD()






