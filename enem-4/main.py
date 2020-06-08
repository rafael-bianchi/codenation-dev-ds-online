# %% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# %% Lendo o dataset de treino
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_student_ids = pd.DataFrame(
    data=test_data['NU_INSCRICAO'], columns=['NU_INSCRICAO'])

# %%
# Separando variável alvo
Y = train_data[['IN_TREINEIRO', 'NU_INSCRICAO']]
# Setando zero para quem faltou/eliminado
Y.fillna(value=0.0, inplace=True)

# Removendo features que só existem no dataset de treino
# e não no dataset de teste
for col in train_data.columns:
    if(col != 'IN_TREINEIRO' and col not in test_data.columns):
        train_data.drop([col], axis=1, inplace=True)

# Removendo features
features_to_remove = ['CO_UF_RESIDENCIA', 'TP_PRESENCA_CH']  # ['NU_INSCRICAO']
train_data.drop(features_to_remove, axis=1, inplace=True)
test_data.drop(features_to_remove, axis=1, inplace=True)

# Criando um único dataset treino+teste, para normalização e hot-encoding
appended_datasets = train_data.append(test_data)



# %% Setando zero para as provas do primeiro dia e do segundo dia, para pessoas que faltaram ou foram eliminados
appended_datasets.loc[(appended_datasets.TP_PRESENCA_CN != 1 ),'NU_NOTA_CN'] = 0.0
appended_datasets.loc[(appended_datasets.TP_PRESENCA_CN != 1), 'NU_NOTA_CH'] = 0.0

appended_datasets.loc[(appended_datasets.TP_PRESENCA_LC != 1 ),'NU_NOTA_LC'] = 0.0
appended_datasets.loc[(appended_datasets.TP_PRESENCA_LC != 1), 'NU_NOTA_MT'] = 0.0


# %% Removendo features com missing values 
for col in appended_datasets.columns[appended_datasets.isnull().any()].tolist():
    if col != 'IN_TREINEIRO':
        appended_datasets.drop([col], axis=1, inplace=True)

# %% hot-encoding para colunas de categoria Questões, SG, CO_ e TP_
category_columns = []

for col in appended_datasets.columns:
    if ('Q0' in col or 'SG_' in col or 'CO_' in col or 'TP_' in col):
        category_columns.append(col) 

appended_datasets = pd.get_dummies(appended_datasets, columns=category_columns, drop_first=True)

# %% Separando treino e teste
train_data = Y[['NU_INSCRICAO']].merge(appended_datasets, how='inner', on=['NU_INSCRICAO'])#pd.merge(Y[['NU_INSCRICAO']], appended_datasets, how='inner', on=['NU_INSCRICAO'])
test_data = test_student_ids[['NU_INSCRICAO']].merge(appended_datasets, how='inner', on=['NU_INSCRICAO'])#pd.merge(test_student_ids[['NU_INSCRICAO']], appended_datasets, how='inner', on=['NU_INSCRICAO'])

train_data.drop(['NU_INSCRICAO'], axis=1, inplace=True)
test_data.drop(['NU_INSCRICAO'], axis=1, inplace=True)

Y = Y['IN_TREINEIRO']
#%% Normalização
scaler = StandardScaler()
train_in_treineiro = train_data['IN_TREINEIRO'].copy()
test_in_treineiro = test_data['IN_TREINEIRO'].copy()
train_data.drop('IN_TREINEIRO', axis=1, inplace=True)
test_data.drop('IN_TREINEIRO', axis=1, inplace=True)

train_data = pd.DataFrame(scaler.fit_transform( train_data), columns=train_data.columns)
test_data =  pd.DataFrame(scaler.fit_transform( test_data ), columns=test_data.columns)

train_data['IN_TREINEIRO'] = train_in_treineiro

# %% Visualizando a distribuição da variável IN_TREINEIRO
plt.figure(figsize=(8, 8))
sns.countplot('IN_TREINEIRO', data=train_data)
plt.title('Balanced Classes - Before Rebalancing')
plt.show()

# %% Rebalanceando o dataset de treino
sm = SMOTE(sampling_strategy='auto')

# Fit the model to generate the data.
oversampled_trainX, oversampled_trainY = sm.fit_sample(train_data.drop('IN_TREINEIRO', axis=1), train_data['IN_TREINEIRO'])
oversampled_train = pd.concat([pd.DataFrame(oversampled_trainX), pd.DataFrame(oversampled_trainY)], axis=1)
oversampled_train.columns = train_data.columns

# %% Visualizando a distribuição da variável IN_TREINEIRO
plt.figure(figsize=(8, 8))
sns.countplot('IN_TREINEIRO', data=oversampled_train)
plt.title('Balanced Classes - After Rebalancing')
plt.show()
Y=oversampled_train['IN_TREINEIRO']
oversampled_train.drop('IN_TREINEIRO', axis=1, inplace=True)

#%% Predição e output
# Parametros do RandomForestRegressor obtidos através do GridSearch
clf = RandomForestClassifier(
            n_estimators=800,
            max_features='auto'
)

clf.fit(oversampled_train, Y)

#%%
pred = clf.predict(test_data)
test_student_ids['IN_TREINEIRO'] = pred

test_student_ids.to_csv('answer.csv', index=False)

# %%
