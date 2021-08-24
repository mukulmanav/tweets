#%%
import pandas as pd
# %%
df=pd.read_csv('text.csv')
# %%
df.head()
# %%
y=df["author"]
# %%
x=df["status"]
# %%
x.isna().sum()
# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,stratify=y)
# %%
from sklearn.feature_extraction.text import CountVectorizer
vector=CountVectorizer(stop_words="english")
x_train=vector.fit_transform(x_train)
# %%
x_test=vector.transform(x_test)
# %%
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train,y_train)
# %%
model.score(x_test,y_test)
# %% PREDICT
tweet="À mesure que les menaces évoluent au pays et dans le monde, on agit pour vous protéger et combattre l’extrémisme violent. Aujourd’hui, le ministre @BillBlair a annoncé qu’on désigne 13 groupes comme entités terroristes au Canada. Plus de renseignements sur ce que ça veut dire "
#%%
data=vector.transform([tweet])
# %%
model.predict(data)
# %%
