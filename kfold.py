import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split

###HYPER_PARAMETER
k = 10          #k_fold
kernel = 'rbf' #kernel
NORM = True    #norm or not


def gen_data():
    feature_path = 'feature.npy'
    label_path = 'label.npy'

    feature = np.load(feature_path)
    label = np.expand_dims(np.load(label_path).astype(float), axis=-1)

    if NORM:
        scaler = StandardScaler()
        feature = scaler.fit_transform(feature)
    #data = np.random.shuffle(np.concatenate((label, feature), axis=-1))
    data = np.concatenate((label, feature), axis=-1)
    np.random.shuffle(data)
    return data

def k_fold(data):
    len = data.shape[0]
    fold_list = [data[int(i*len/10):int((i+1)*len/10)] for i in range(10)]
    return fold_list

def svm(fold_data):
    svc = SVC(kernel=kernel, class_weight='balanced',)
    ##find the best c and gamma
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    n_iter_search = 20
    param_grid = {'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}
    grid = RandomizedSearchCV(svc, param_grid, n_iter=n_iter_search, n_jobs=-1)

    total_ac = 0.

    for test_ind in range(k):
        print('fold{} starts....'.format(test_ind), )
        List = []
        for i in range(k):
            if i!=test_ind:
                List.append(fold_data[i])
        print('fns1')
        train_set = np.concatenate(tuple(List), axis=0)
        test_set = fold_data[test_ind]
        train_x = train_set[:, 1:]
        train_y = train_set[:, 0].astype(int)

        test_x = test_set[:, 1:]
        test_y = test_set[:, 0].astype(int)
        grid.fit(train_x, train_y)
        print('fns2')
        score = grid.score(test_x, test_y)
        print('ac={}'.format(score))
        total_ac += score
    ac = total_ac / k
    print(ac)

def train():
    svm(k_fold(gen_data()))
if __name__ == '__main__':
    train()
