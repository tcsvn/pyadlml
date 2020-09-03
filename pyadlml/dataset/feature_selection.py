
# compute a chi-squared dependence table
def rep2onoff(rep):
    df = rep.copy().iloc[:,:-1] # get raw without activities
    df = df.reset_index(drop=True)
    df = df.astype(int)

    for device in df.columns:
        mask1 = df[[device]] == 1
        col_off = device + ' Off'
        df[[col_off]] = df[[device]].astype(bool)
        df[[col_off]] = ~df[[col_off]]
        df[[col_off]] = df[[col_off]].astype(int)
        df = df.rename(columns={device: device + ' On'})
    df = df.astype(bool)
    return df

def chi2(X, ):
    # sklearn feature selection chi2
    from sklearn.feature_selection import chi2
    rep = raw
    df = rep2onoff(rep)
    chi2stats = chi2(df.values, raw['activity'])
    index = [item + ' On' for item in rep.columns[:-1]]
    index += [item + ' Off' for item in rep.columns[:-1]]
    df = pd.DataFrame(data={'p-value':chi2stats[1], 'chi2 stat': chi2stats[0]}, index=index)
    df = df.sort_index()
    return df


def mut_inf_01(rep):
    # sklearn feature selection chi2
    from sklearn.feature_selection import mutual_info_classif as mic
    rep = raw
    df = rep2onoff(rep)
    X, y = df.values, raw['activity']
    
    mistats = mic(X, y, n_neighbors=4)
    index = [item + ' On' for item in rep.columns[:-1]]
    index += [item + ' Off' for item in rep.columns[:-1]]
    
    df = pd.DataFrame(data={'mutual information':mistats}, index=index)
    df = df.sort_index()
    return df

from sklearn.feature_selection import mutual_info_classif as mic

def mut_inf(rep):
    col_name = 'Mutual Information'
    X, y = rep.values[:,:-1], rep.values[:,-1:].squeeze()
    mistats = mic(X, y, n_neighbors=4)
    df = pd.DataFrame(data={col_name : mistats}, index=rep.columns[:-1])
    df = df.sort_values(by=col_name, ascending=False)
    return df