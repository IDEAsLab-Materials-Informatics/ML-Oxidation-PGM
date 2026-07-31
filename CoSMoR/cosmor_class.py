import pandas as pd
import numpy as np
import pickle
import os
from core_functions import (create_alloys, create_features, make_predictions,
                            calculate_dY_dX, calculate_delta_X, calculate_feat_contributions)

SEP = os.sep

class cosmor:
    
    def __init__(self):
        
        print("\n--- Collecting user input to create 'CoSMoR' class instance ---")

        self.A = str(input("\t-Enter component A (e.g.:'Al', 'AlTi', 'Al2Ti'):\t"))
        self.B = str(input("\t-Enter component B (e.g.:'Co', 'CoCr', 'Co2Cr'):\t"))
        self.dc = float(input("\t-Enter composition step size in at. fraction (typically 0.01):\t"))
        self.cAmin = float(input("\t-Enter starting concentration of component A:\t"))
        self.cAmax = float(input("\t-Enter maximum concentration of component A:\t"))
        self.replace_el = input("\t-Replace element in B (e.g. 'Ni', or leave blank for proportional dilution):\t").strip() or None
        self.dX = float(input("\t-Enter feature step size (typically 0.02):\t"))
        self.time_ip = float(input("\t-Enter Time (typically in hrs):\t"))
        self.time_ip = (self.time_ip - 0.5)/(500 - 0.5)
        self.save_bool = input("\t-Save results? [Yes/No (Y/N)]:\t")
        self.plot_bool = input("\t-Plot results? [Yes/No (Y/N)]:\t")
        print("CoSMoR instance created successfully.")
    
            
    def save_cosmor(self):
        
        results_dir = f"cosmor_results{SEP}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        savename = f"{self.A}-{self.B}-dc_{self.dc}-dX_{self.dX}"
        print(f"|Saving cosmor results as excel file '{savename}.xlsx'...", end=" ", flush=True)
        
        writer = pd.ExcelWriter(f"{results_dir}{savename}.xlsx", engine='xlsxwriter')
        
        df_Y_pred = pd.DataFrame(data=self.Y_pred_ML, columns=["Y_predicted"])
        pd.concat([self.df_alloys, df_Y_pred], axis=1).to_excel(writer, sheet_name="Y_Prediction", index = True)
        pd.concat([self.df_alloys, self.df_x_feats], axis=1).to_excel(writer, sheet_name="Feature_values", index = True)
        pd.concat([self.df_alloys, self.delta_X], axis=1).to_excel(writer, sheet_name="delta_X", index = True)
        pd.concat([self.df_alloys, self.dY_dX], axis=1).to_excel(writer, sheet_name="PLD-dY_dX", index = True)
        pd.concat([self.df_alloys, self.loc_feat_contri], axis=1).to_excel(writer, sheet_name="loc_feat_contributions", index = True)
        pd.concat([self.df_alloys, self.cum_feat_contri], axis=1).to_excel(writer, sheet_name="cum_feat_contributions", index = True)
        
        writer.close()
        print("DONE.")
        
        print(f"|Saving cosmor class as pickle file '{savename}.cosmor'...", end=" ", flush=True)
        with open(f"{results_dir}{savename}.cosmor", 'wb') as f:
            pickle.dump(self, f)
        print("DONE.")
    
    
    def plot_cosmor(self):
        
        import matplotlib
        from matplotlib import pyplot as plt
        
        print(f"|Creating plots for CoSMoR results...", end=" ", flush=True)
        x_axis = np.array(self.df_alloys[f"x [A={self.A}]"])
        x_axis_label = f"x [({self.A}) at.fraction]"
        
        font = {'family' : 'DejaVu Sans',
                'weight' : 'regular',
                'size'   : 20}
        matplotlib.rc('font', **font)

        fig_size = (8, 6)
        title_fs, label_fs, legend_fs = 20, 18, 15
        ss = 60 #symbol size
        lw = 5 #linewidth
        nCols = 2
        nRows = 2

        results_dir = f"cosmor_results{SEP}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        fig_savename = f"{self.A}-{self.B}-dc_{self.dc}-dX_{self.dX}"

        # --- Combined figure (2x2) with global legend outside ---
        fig, axes = plt.subplots(nRows, nCols,
                                 figsize=(fig_size[0]*nCols, fig_size[1]*nRows + 2))

        subplot_titles = [
            "(a) Cumulative feature contributions",
            "(b) Local feature contributions (For each composition step)",
            "(c) Comparing Y prediction (ML vs. CoSMoR)",
            "(d) Feature variations",
        ]
        for ax, title in zip(axes.flat, subplot_titles):
            ax.set_title(title, fontsize=title_fs)
            ax.set_xlabel(x_axis_label, fontsize=label_fs)

        # Subplot 1: Cumulative feature contributions
        ax1 = axes[0, 0]
        ax1.set_ylabel("Feature contributions", fontsize=label_fs)
        for feat in self.feat_name_list:
            ax1.plot(x_axis, self.cum_feat_contri[feat], linewidth=lw, label=feat, alpha=0.75)
        ax1.plot([x_axis[0], x_axis[-1]], [self.Y_baseline_value, self.Y_baseline_value],
                 "--", linewidth=2, label="baseline", alpha=1)
        ax1.plot(x_axis, self.Y_pred_ML, "--", linewidth=lw-1, label="Y_overall", color='#BCBD22', alpha=1.0)

        # Subplot 2: Local feature contributions
        ax2 = axes[0, 1]
        ax2.set_ylabel("Feature contributions", fontsize=label_fs)
        for feat in self.feat_name_list:
            ax2.plot(x_axis[1:], np.array(self.loc_feat_contri[feat])[1:],
                     label=feat, alpha=0.75)

        # Subplot 3: ML vs CoSMoR predictions
        ax3 = axes[1, 0]
        ax3.set_ylabel("Overall Y prediction", fontsize=label_fs)
        ax3.plot(x_axis, self.Y_pred_ML, "--", linewidth=lw, label="Y_ML", alpha=0.75)
        ax3.plot(x_axis, self.Y_pred_cosmor, linewidth=lw, label="Y_CoSMoR", alpha=0.75)

        # Subplot 4: Feature variations
        ax4 = axes[1, 1]
        ax4.set_ylabel("Feature values", fontsize=label_fs)
        for feat in self.feat_name_list:
            ax4.plot(x_axis, self.df_x_feats[feat], linewidth=lw, label=feat, alpha=0.75)

        # --- Global legend: collect unique handles/labels from all axes ---
        all_handles, all_labels = [], []
        seen_labels = set()
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen_labels:
                    all_handles.append(h)
                    all_labels.append(l)
                    seen_labels.add(l)

        fig.legend(all_handles, all_labels,
                   loc='upper center',
                   ncol=min(len(all_labels), 6),
                   fontsize=legend_fs,
                   bbox_to_anchor=(0.5, 1.0),
                   frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave room at top for legend
        print("DONE.")

        # Save combined figure as PDF and PNG
        print(f"|Saving combined plots as pdf '{fig_savename}-PLOTS.pdf'...", end=" ", flush=True)
        fig.savefig(f"{results_dir}{fig_savename}-PLOTS.pdf", bbox_inches='tight')
        print("DONE.")
        print(f"|Saving combined plots as png '{fig_savename}-PLOTS.png'...", end=" ", flush=True)
        fig.savefig(f"{results_dir}{fig_savename}-PLOTS.png", dpi=150, bbox_inches='tight')
        print("DONE.")
        plt.close(fig)

        # --- Save each subplot as an individual PNG ---
        subplot_info = [
            ("cum_feat_contributions",  self._make_subplot_cum_feat,   "Cumulative feature contributions",                    "Feature contributions"),
            ("loc_feat_contributions",  self._make_subplot_loc_feat,   "Local feature contributions (For each composition step)", "Feature contributions"),
            ("Y_prediction_comparison", self._make_subplot_Y_pred,     "Comparing Y prediction (ML vs. CoSMoR)",              "Overall Y prediction"),
            ("feature_variations",      self._make_subplot_feat_var,   "Feature variations",                                  "Feature values"),
        ]
        print(f"|Saving individual subplot PNGs...", end=" ", flush=True)
        for tag, plot_fn, title, ylabel in subplot_info:
            fig_s, ax_s = plt.subplots(figsize=fig_size)
            ax_s.set_title(title, fontsize=title_fs)
            ax_s.set_xlabel(x_axis_label, fontsize=label_fs)
            ax_s.set_ylabel(ylabel, fontsize=label_fs)
            plot_fn(ax_s, x_axis, lw)
            fig_s.tight_layout()
            fig_s.savefig(f"{results_dir}{fig_savename}-{tag}.png", dpi=150, bbox_inches='tight')
            plt.close(fig_s)
        print("DONE.")

        # --- Save standalone legend PNG ---
        print(f"|Saving standalone legend as png '{fig_savename}-LEGEND.png'...", end=" ", flush=True)
        fig_leg = plt.figure(figsize=(max(4, len(all_labels) * 1.2), 1.2))
        fig_leg.legend(all_handles, all_labels,
                       loc='center',
                       ncol=1,
                       fontsize=legend_fs,
                       frameon=True)
        fig_leg.savefig(f"{results_dir}{fig_savename}-LEGEND.png", dpi=150, bbox_inches='tight')
        plt.close(fig_leg)
        print("DONE.")

    # --- Private helpers used by individual subplot saves ---

    def _make_subplot_cum_feat(self, ax, x_axis, lw):
        import matplotlib.pyplot as plt
        for feat in self.feat_name_list:
            ax.plot(x_axis, self.cum_feat_contri[feat], linewidth=lw, label=feat, alpha=0.75)
        ax.plot([x_axis[0], x_axis[-1]], [self.Y_baseline_value, self.Y_baseline_value],
                "--", linewidth=2, label="baseline", alpha=1)
        ax.plot(x_axis, self.Y_pred_ML, "--", linewidth=lw-2, label="Y_overall", color='black', alpha=1.0)

    def _make_subplot_loc_feat(self, ax, x_axis, lw):
        for feat in self.feat_name_list:
            ax.plot(x_axis[1:], np.array(self.loc_feat_contri[feat])[1:],
                    label=feat, alpha=0.75)

    def _make_subplot_Y_pred(self, ax, x_axis, lw):
        ax.plot(x_axis, self.Y_pred_ML, "--", linewidth=lw, label="Y_ML", alpha=0.75)
        ax.plot(x_axis, self.Y_pred_cosmor, linewidth=lw, label="Y_CoSMoR", alpha=0.75)

    def _make_subplot_feat_var(self, ax, x_axis, lw):
        for feat in self.feat_name_list:
            ax.plot(x_axis, self.df_x_feats[feat], linewidth=lw, label=feat, alpha=0.75)
        
    
    def run_cosmor(self):
        
        print("\n--- Running 'CoSMoR' framework ---")
        print("|Creating alloys along composition pathway...", end=" ", flush=True)
        self.df_alloys = create_alloys(self.A, self.B, self.dc, self.cAmin, self.cAmax)
        print("DONE.")
        
        print("|Creating feature values...", end=" ", flush=True)
        self.df_x_feats = create_features(self.df_alloys)
        self.df_x_feats["Time"] = np.full((len(self.df_x_feats)), self.time_ip)
        print("DONE.")
        
        print("|Extracting feature names...", end=" ", flush=True)
        self.feat_name_list = self.df_x_feats.columns.to_list()
        print("DONE.")
        
        print("|Generating overall model predictions along composition pathway...", end=" ", flush=True)
        self.Y_pred_ML = make_predictions(self.df_x_feats)
        print("DONE.")
        
        print("|Identifying baseline value of target property...", end=" ", flush=True)
        self.Y_baseline_value = self.Y_pred_ML[0]
        print("DONE.")
        
        print("|Calculating local partial dependencies [dY/d(Xi)] along the composition pathway...")
        self.dY_dX = calculate_dY_dX(self.df_x_feats, self.dX)

        print("|Calculating feature variations for compositional stimulus along composition pathway...", end=" ", flush=True)
        self.delta_X = calculate_delta_X(self.df_x_feats)
        print("DONE.")
        
        print("|Calculating feature contributions along composition pathway...", end=" ", flush=True)
        
        self.loc_feat_contri, self.cum_feat_contri = calculate_feat_contributions(self.Y_baseline_value,
                                                                                  self.df_x_feats,
                                                                                  self.dY_dX,
                                                                                  self.delta_X)
        
        print("DONE.")
        
        print("|Calculating predictions based on cumulative contributions...", end=" ", flush=True)
        self.Y_pred_cosmor = self.Y_baseline_value + np.sum(self.cum_feat_contri -
                                                            self.Y_baseline_value, axis=1)
        
        print("DONE.")
        
        
        if self.save_bool.lower() in ["yes", "y"]:
            self.save_cosmor()
            
        if self.plot_bool.lower() in ["yes", "y"]:
            self.plot_cosmor()
            
        print("\n--- COMPLETED ---")
