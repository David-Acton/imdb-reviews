import pandas as pd

df_0 = pd.read_csv("data/dataset_3.csv", index_col=0, encoding='windows-1254')
df_1 = pd.read_csv("data/dataset_2.csv", index_col=0, encoding='windows-1254')
df_2 = pd.read_csv("data/dataset_1.csv", index_col=0)

df = pd.concat([df_0, df_1, df_2])
df_g = df[df.isin(['g']).any(axis=1)]
df_b = df[df.isin(['b']).any(axis=1)]

bad_file = open("bad_reviews.txt", "w", encoding="utf-8")
good_file = open("good_reviews.txt", "w", encoding="utf-8")

for short, full in zip(df_g['short_review'], df_g['full_review']):
    good_file.write(short.strip() + ' ' + "".join(full.splitlines()) + "\n")

for short, full in zip(df_b['short_review'], df_b['full_review']):
    bad_file.write(short.strip() + ' ' + "".join(full.splitlines()) + "\n")

bad_file.close()
good_file.close()