from downloaders import download_reviews 
import pandas as pd

data = download_reviews.get_reviews("tt0099674")
df = pd.DataFrame.from_dict(data)
df.to_csv("data.csv")

print(df)