import requests
import pandas as pd
pd.set_option('display.max_colwidth', None)

url = "https://api.liquipedia.net/api/v1/match"

payload = {
    "wiki": "fighters",
    "query": "date, opponent1score, opponent2score, game, matchid, pagename, objectname",
    "conditions": "[[walkover::!1]] AND [[walkover::!2]] AND [[mode::singles]] AND [[opponent1::!Bye]] AND [[opponent2::!Bye]]",
    "order": "date ASC, objectname ASC",
    "limit": "1000",
    "apikey": "P2uJOrC8DKbiSNl5CQKmZNI2non7fX2WUpKUeaveJiZ6mD6ILLKtXasiE137JDMScUnajPpYQXHd9c8o1Fe62M2QhohiucU4xJ01fIubVNnAthH55ciKrEAgQuyIYi9G",
    "offset": "280000"
}

response = requests.post(url, data=payload)

data = response.json()['result']
df = pd.DataFrame(data)
print(len(df))
print(df.columns)
# print(df.groupby('pagename').size())

# print(df.loc[df['objectname'].str.len().nlargest(50).index]['objectname'])

print(df[df['objectname'].str.contains("Nafutaren", na=False)]['pagename'])



