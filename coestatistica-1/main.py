# %% Imports
import pandas as pd

# %% Reading file
df = pd.read_csv('desafio1.csv')


# %% Agregando e renomeando columan
agg = df.groupby('estado_residencia')['pontuacao_credito'] \
        .agg([pd.Series.mode,
              pd.Series.median,
              pd.Series.mean,
              pd.Series.std])

agg.columns = ['moda', 'mediana', 'media', 'desvio_padrao']

# %% Transpose e salvar output
agg.T.to_json('submission.json')
