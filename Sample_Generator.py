"""
Running this code generates a random sampling from X1-7
Generates labels from the function (fn)
Creates training and test sets
* Exports:
    dtrain.pickle and dtest.pickle
    picked out -> can pickled in to other files
* Requires:
    fn.py in this directory
"""

def generate():

    import numpy as np
    import pandas as pd
    import pickle
    import seaborn as sns
    from fn import eqn  # requires fn.py in this dir

    # Define means and std of all 7 Normal RVs
    m1, s1 = 350, 35
    m2, s2 = 50.8, 5.08
    m3, s3 = 3.81, 0.381
    m4, s4 = 173, 17.3
    m5, s5 = 9.38, 0.938
    m6, s6 = 33.1, 3.31
    m7, s7 = 0.36, 0.0036

    # Generate sample
    ss = 10000  # Sample size
    x1 = np.random.normal(m1, s1, ss)
    x2 = np.random.normal(m2, s2, ss)
    x3 = np.random.normal(m3, s3, ss)
    x4 = np.random.normal(m4, s4, ss)
    x5 = np.random.normal(m5, s5, ss)
    x6 = np.random.normal(m6, s6, ss)
    x7 = np.random.normal(m7, s7, ss)

    # Generate Labels
    l = []
    for i in range(0, ss):
        y = eqn(x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i])
        l.append(y)
    #print(l)  # quick test

    # create dataset as pd dataframe
    ds = pd.DataFrame(l, columns = ['label'])
    ds['X1'] = x1
    ds['X2'] = x2
    ds['X3'] = x3
    ds['X4'] = x4
    ds['X5'] = x5
    ds['X6'] = x6
    ds['X7'] = x7
    #print(ds.head())   #quick test

    # create training and testing ds
    d_train = ds.sample(frac=0.8,random_state=0)
    d_test = ds.drop(d_train.index)
   # print(d_test.head())

    # Export (pickle out) datasets to files
    with open('dtrain.pickle', 'wb') as file:
        pickle.dump(d_train, file)

    with open('dtest.pickle', 'wb') as file:
        pickle.dump(d_test, file)

    sns.pairplot(d_train, diag_kind="kde")


if __name__ == '__main__':
    generate()
