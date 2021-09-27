import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
 
pd.options.mode.chained_assignment = None

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
r = requests.get(url)
json = r.json()

elements_df = pd.DataFrame(json['elements'])
elements_types_df = pd.DataFrame(json['element_types'])
teams_df = pd.DataFrame(json['teams'])

df = elements_df[['second_name','team','element_type','selected_by_percent','now_cost','form','value_season','total_points']]

df['position'] = df['element_type'].map(elements_types_df.set_index('id').singular_name_short)
df['team'] = df['team'].map(teams_df.set_index('id').name)
df['value'] = df['value_season'].astype(float)
df['selected_by_percent'] = df['selected_by_percent'].astype(float)

df['now_cost'] = df['now_cost'].astype(float).div(10)
df = df.drop(columns=['element_type', 'value_season'])

pivot = df.pivot_table(index='position', values='value', aggfunc=np.mean).reset_index()
team_pivot = df.pivot_table(index='team',values='value',aggfunc=np.mean).reset_index()

scaler = MinMaxScaler(feature_range=(0,10))
scaled_df = pd.DataFrame(df, columns=['form','total_points','value'])
df_scaled = pd.DataFrame(scaler.fit_transform(scaled_df), columns=['form','total_points','value'])

df['overall'] = round((df_scaled['form'] + df_scaled['total_points'] + 2 * df_scaled['value'])/4,2)
df = df.loc[df['value'] > 0]

gkp_df = df.loc[df['position'] == 'GKP'].sort_values('overall', ascending=False)
def_df = df.loc[df['position'] == 'DEF'].sort_values('overall', ascending=False)
mid_df = df.loc[df['position'] == 'MID'].sort_values('overall', ascending=False)
fwd_df = df.loc[df['position'] == 'FWD'].sort_values('overall', ascending=False)

print("Fantasy Premier League Analytics Report:", datetime.today().strftime('%Y-%m-%d'))
print("=" * 120, "\n")

print("Top GKP (Overall)")
print("-" * 120)
print(gkp_df.head(10).to_string(index=False))
print("-" * 120, "\n")

print("Top DEF (Overall)")
print("-" * 120)
print(def_df.head(10).to_string(index=False))
print("-" * 120, "\n")

print("Top MID (Overall)")
print("-" * 120)
print(mid_df.head(10).to_string(index=False))
print("-" * 120, "\n")

print("Top FWD (Overall)")
print("-" * 120)
print(fwd_df.head(10).to_string(index=False))
print("-" * 120, "\n")
