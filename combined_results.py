import pandas as pd

df1 = pd.read_csv("DT_results.csv", header=0)
df2 = pd.read_csv("DT1_results.csv", header=0)

frames = [df1, df2]
f_df = pd.concat(frames)
f_df.reset_index(inplace=True, drop=True)

f_df.to_csv("Final_Results_DT.csv")


print(f_df)
