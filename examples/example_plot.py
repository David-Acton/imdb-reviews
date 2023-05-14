
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.pyplot import figure


#Makes a plot based on ratings over time. 



df = pd.read_csv('data.csv')

df_review_dates = list(df.review_date.values) 
df_datetime_format = pd.to_datetime(df_review_dates, format='%d %B %Y')

figure(figsize=(30, 10), dpi=80)
x=df_datetime_format
y=df.rating_value
color = np.where(y < 6, "red", "green")

plt.title('Captain Marvel', fontsize = 30, color = "red")
plt.xlabel("Time", fontsize = 15)
plt.ylabel("Ratings", fontsize = 15)
plt.scatter(x, y,marker=".", c = color, s =2)
