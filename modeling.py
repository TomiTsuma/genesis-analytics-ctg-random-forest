from kennard_stone import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from ray import train, tune, data, serve
from imblearn.over_sampling import SMOTE

def splitData(X,y):
    X_train, X_validation_test, y_train, y_validation_test = train_test_split(X, y, test_size=0.3)
    X_valid, X_test, y_valid, y_test = train_test_split(X_validation_test, y_validation_test, test_size=0.6)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def hyperparamTuning(X_valid, X_test, y_valid, y_test):
    search_space = {
    "n_estimators": tune.choice([50, 100, 150, 200, 250]),
    "max_depth": tune.choice([None, 10, 20, 30, 40 ,50, 60]),
    "min_samples_split": tune.choice([2, 5, 10, 12,15,20,32,36,40]),
    "criterion": tune.choice(["gini", "entropy", "log_loss"]),
    "min_samples_leaf": tune.randint(1, 15),
    "max_features": tune.choice(["sqrt", "log2", None]),
    "X_valid": X_valid,
    "X_test": X_test,
    "y_valid": y_valid,
    "y_test": y_test,
}

    analysis = tune.run(
        trainRF,
        config=search_space,
        metric="accuracy",
        mode="max",
        num_samples=200,
    )

    

    best_config = analysis.get_best_config(metric="accuracy", mode="max")

    del best_config['X_valid']
    del best_config['y_valid']
    del best_config['X_test']
    del best_config['y_test']

    return best_config

def trainRF(config):
    valid_features = config['X_valid']
    valid_target = config['y_valid']
    test_features = config['X_test']
    test_target = config['y_test']
    del config['X_valid']
    del config['y_valid']
    del config['X_test']
    del config['y_test']
    model = RandomForestClassifier(**config)
    model.fit(valid_features, valid_target)
    score = model.score(test_features, test_target)
    return {'accuracy': score}


def trainRFClassifier(best_config, X_train, y_train):
    rf = RandomForestClassifier(**best_config)
    rf.fit(X_train, y_train)

    pickle.dump(rf, open("model/RandomForestClassifier.pkl","wb"))

    return rf

def overSample(X_train, y_train):
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    return X_train, y_train                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

    