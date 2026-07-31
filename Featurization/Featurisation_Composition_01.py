import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from CBFV import composition
import seaborn as sns

from f_helpers import get_miedema_enthalpy
from f_helpers2 import get_asymmetry

import os
SEP = os.sep

df = pd.read_excel("Data_collected.xlsx")
print("df shape: {df.shape}")
df.head()

df_cbfv = df[["alloy_name","Mass_Change"]].copy()
df_cbfv.columns = ["formula","target"]
print(df_cbfv.head())

x_cbfv, y_cbfv, formula_cbfv, _ = composition.generate_features(df_cbfv, elem_prop = 'oliynyk')
print(x_cbfv.head)

feats_from_cbfv = ["avg_Melting_point_(K)"]
x_feats = x_cbfv.copy()[feats_from_cbfv]/1423

H_mied = np.array([get_miedema_enthalpy(alloy) for alloy in df["alloy_name"]])

alloy_name_list = df["alloy_name"]
# x_feats["r_asym"] = get_asymmetry(alloy_name_list, feat_key = "r_cov")



x_feats["Time"] = df["Time"]

x_feats["Temp"] = df["Temp"]
x_feats["Ni"] = df["Ni"]
x_feats["Al"] = df["Al"]
x_feats["Pt"] = df["Pt"]
x_feats["Pd"] = df["Pd"]
x_feats["Ir"] = df["Ir"]
x_feats["Rh"] = df["Rh"]


x_feats.head()
x_feats.to_csv("db_PGM_CMP_01_feature.csv")
print(x_feats.head())

xmin = np.amin(x_feats, axis = 0);
xmax = np.amax(x_feats, axis = 0);
xmin.to_csv("xmin_db_PGM_CMP_01.csv")
xmax.to_csv("xmax_db_PGM_CMP_01.csv")

x_feats_norm = (x_feats - xmin)/(xmax - xmin)
print(x_feats_norm.head())

df_with_norm_feats = x_feats_norm.copy()
df_with_norm_feats = pd.concat([df[["alloy_name","Mass_Change"]], df_with_norm_feats], axis = 1)

corr_mat = df_with_norm_feats.drop(["alloy_name","Time","Temp"], axis = 1).corr(method="pearson")
sns.heatmap(corr_mat, annot = True)
df_with_norm_feats.to_excel("PGM_CMP_01_norm_feature.xlsx", index = False)

df_to_plot = df_with_norm_feats.drop(["alloy_name"], axis = 1)
sns.pairplot(df_to_plot)
