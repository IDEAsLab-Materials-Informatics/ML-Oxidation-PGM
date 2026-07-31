import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from CBFV import composition
import seaborn as sns
from f_helpers import get_miedema_enthalpy
from f_helpers2 import get_asymmetry, get_comp_avg
import os
SEP = os.sep
df = pd.read_excel("Data_collected_PGM.xlsx")
print(f"df shape: {df.shape}")
df.head()
df_cbfv = df[["alloy_name","Mass_Change"]].copy()
df_cbfv.columns = ["formula","target"]
print(df_cbfv.head())
x_cbfv, y_cbfv, formula_cbfv, _ = composition.generate_features(df_cbfv, elem_prop = 'oliynyk')
print(x_cbfv.head())
feats_from_cbfv = []
x_feats = x_cbfv.copy()[feats_from_cbfv]
x_feats.head()
H_mied = np.array([get_miedema_enthalpy(alloy) for alloy in df["alloy_name"]])
alloy_name_list = df["alloy_name"]
x_feats["r_asym"] = get_asymmetry(alloy_name_list, feat_key = "r_cov")
x_feats["avg_number_of_valence_electrons"] = get_comp_avg(alloy_name_list, feat_key="VEC")
x_feats["Coh_E"] = get_comp_avg(alloy_name_list, feat_key = "Coh_E")
x_feats["EN_Pauling"] = get_asymmetry(alloy_name_list, feat_key = "EN_Pauling")
x_feats["H_chem"] = H_mied[:,0]
x_feats["H_el"] = H_mied[:,1]
x_feats["Time"] = df["Time"]
x_feats["Temp"] = df["Temp"]
x_feats.to_csv("db_PGM_CBFV_01_feats.csv")
print(x_feats.head())


xmin = np.amin(x_feats, axis = 0);
xmin.loc["r_asym"] = 0
xmin.loc["H_el"] = 0
xmax = np.amax(x_feats, axis = 0);

xmin.to_csv("xmin_db_PGM_CBFV_01.csv")
xmax.to_csv("xmax_db_PGM_CBFV_01.csv")

x_feats_norm = (x_feats - xmin)/(xmax - xmin)
print(x_feats_norm.head())
df_with_norm_feats = x_feats_norm.copy()
df_with_norm_feats = pd.concat([df[["alloy_name","Mass_Change"]], df_with_norm_feats], axis = 1)
corr_mat = df_with_norm_feats.drop(["alloy_name","Time","Temp"], axis = 1).corr(method="pearson")
sns.heatmap(corr_mat, annot = True)
df_with_norm_feats.to_excel("PGM_CBFV_01_norm_feature.xlsx", index = False)
df_to_plot = df_with_norm_feats.drop(["alloy_name"], axis = 1)
sns.pairplot(df_to_plot)