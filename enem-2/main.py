# %% Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# %% Lendo o dataset de treino
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_student_ids = pd.DataFrame(
    data=test_data['NU_INSCRICAO'], columns=['NU_INSCRICAO'])

# %%
# Separando variável alvo
Y = train_data[['NU_NOTA_MT', 'NU_INSCRICAO']]
# Setando zero para quem faltou/eliminado
Y.fillna(value=0.0, inplace=True)

# Removendo features que só existem no dataset de treino
# e não no dataset de teste
for col in train_data.columns:
    if(col not in test_data.columns):
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
    appended_datasets.drop([col], axis=1, inplace=True)

# %% hot-encoding para colunas de categoria Questões, SG, CO_ e TP_
category_columns = []

for col in appended_datasets.columns:
    if ('Q0' in col or 'SG_' in col or 'CO_' in col or 'TP_' in col):
        category_columns.append(col) 

appended_datasets = pd.get_dummies(appended_datasets, columns=category_columns, drop_first=True)

# %% Separando treino e teste
train_data = pd.merge(Y[['NU_INSCRICAO']], appended_datasets, how='inner', on=['NU_INSCRICAO'])
test_data = pd.merge(test_student_ids[['NU_INSCRICAO']], appended_datasets, how='inner', on=['NU_INSCRICAO'])

train_data.drop(['NU_INSCRICAO'], axis=1, inplace=True)
test_data.drop(['NU_INSCRICAO'], axis=1, inplace=True)

Y = Y['NU_NOTA_MT']
#%% Normalização
scaler = StandardScaler()
train_data = scaler.fit_transform( train_data )
test_data = scaler.transform( test_data )

#%% Predição e output
# Parametros do RandomForestRegressor obtidos através do GridSearch
regressor = RandomForestRegressor(
            n_estimators=800,
            max_features='auto'
)

regressor.fit(train_data, Y)
pred = regressor.predict(test_data)
test_student_ids['NU_NOTA_MT'] = pred

test_student_ids.to_csv('answer.csv', index=False)