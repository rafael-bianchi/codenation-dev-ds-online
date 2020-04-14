#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[21]:


import pandas as pd
import numpy as np


# In[22]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[37]:


black_friday['User_ID'][(black_friday.Gender == 'F') & (black_friday.Age == '26-35')].shape[0]


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[24]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[25]:


def q2():
    #return black_friday['User_ID'][(black_friday.Gender == 'F') & (black_friday.Age == '26-35')].unique().shape[0]
    return black_friday['User_ID'][(black_friday.Gender == 'F') & (black_friday.Age == '26-35')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[26]:


def q3():
    return black_friday['User_ID'].unique().shape[0]


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[27]:


def q4():
    return black_friday.dtypes.unique().shape[0]


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[38]:


def q5():
    aux = black_friday.dropna(axis=0)

    return round(1 - (len(aux) / len(black_friday)),3)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[29]:


def q6():
    return max(black_friday.isna().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[30]:


def q7():
    return black_friday['Product_Category_3'].mode()[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[31]:


def q8():
    p = black_friday['Purchase']
    n=(p-p.min())/(p.max()-p.min())
    return n.mean().item()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[32]:


def q9():
    p = black_friday['Purchase']
    stdrized=(p-p.mean())/p.std()
    return stdrized[(stdrized >= -1) & (stdrized <= 1)].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[59]:


def q10():
    #filtra todos os prd cat 2 que são N/A
    p_cat_2_na = black_friday[black_friday['Product_Category_2'].isnull()]

    #retorna se todos os prd cat 3 são N/A também
    return p_cat_2_na['Product_Category_3'].isna().all().item()

