import os
os.chdir(r"C:\nlp-ml-journey\preparation\books\Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.neighbors



# preare the data 
def prepare_country_stats(oecd_bli, gdp_per_capita):
    # keep only Life satisfaction and the 'Total' inequality slice
    oecd_bli = oecd_bli[
        (oecd_bli["Indicator"] == "Life satisfaction") &
        (oecd_bli["Inequality"] == "Total")
    ]

    # pivot so countries are rows, indicator is a column
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    # rename GDP column, set country as index
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    # merge
    full_country_stats = pd.merge(
        left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True
    )

    # sort and remove outliers (to match the book)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))

    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[keep_indices]


# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# visualize the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

# select a linear model
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# train the model
model.fit(X, y)

# make a prediction for Cyprus
X_new = np.array([[22587.0]], dtype=float) # cyprus's GDP per capita
model.predict((X_new))