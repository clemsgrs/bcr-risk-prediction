import pandas as pd
import os

def generate_pairwise_latex(pairwise_dir, cohorts, train_setting="RUMC"):
    """
    Parses pairwise comparison results CSVs and generates LaTeX code for a summary table.
    """
    label = train_setting.upper()
    if "+" not in label:
        label += "-only"

    comparisons = [
        ("Prost40M", "UNI"),
        ("Prost40M", "Virchow2"),
        ("Prost40M", "H-optimus-0"),
        ("UNI", "Virchow2"),
        ("UNI", "H-optimus-0"),
        ("Virchow2", "H-optimus-0"),
    ]

    header = r"""\begin{table}[ht]
		\centering
		\caption{\textbf{Pairwise comparison of deep learning models trained on \texttt{""" + label + r"""} data.} The table reports the difference in concordance index ($\Delta$) between pairs of encoders when evaluated on each test cohort. For each comparison, $\Delta$ values are shown together with 95\% bootstrap confidence intervals (CI) and false discovery rate (FDR) adjusted $q$ values. Positive $\Delta$ values indicate higher performance of the encoder listed first in the comparison, whereas negative values indicate higher performance of the encoder listed second. Values that remain significant after correction ($q < 0.05$) are shown in bold.}
		\resizebox{\textwidth}{!}{\begin{tabular}{llccc}
				\toprule
				\textbf{Cohort} & \textbf{Model Comparison} & $\Delta$ & \textbf{95\% CI} & $q$ \\
				\midrule"""

    footer = r"""				\bottomrule                                 
			\end{tabular}                                
		}                                             
		\label{table:pairwise-""" + train_setting.lower() + r"""}                   
	\end{table}"""

    latex_rows = []
    
    for i, cohort in enumerate(cohorts):
        file_path = os.path.join(pairwise_dir, f"pairwise-{cohort}-{train_setting}.csv")
        if not os.path.exists(file_path):
            print(f"% Warning: {file_path} does not exist.")
            continue
            
        df = pd.read_csv(file_path)
        
        for j, (a, b) in enumerate(comparisons):
            # Try both orders
            mask = (df['encoder_a'] == a) & (df['encoder_b'] == b)
            flip = False
            if not mask.any():
                mask = (df['encoder_a'] == b) & (df['encoder_b'] == a)
                flip = True
                
            if not mask.any():
                print(f"% Warning: comparison {a} vs {b} not found in {file_path}")
                continue
                
            row = df[mask].iloc[0]
            delta = row['delta']
            ci_low = row['ci_low']
            ci_high = row['ci_high']
            q = row['q']
            
            if flip:
                delta = -delta
                ci_low, ci_high = -ci_high, -ci_low
            
            # Format delta
            delta_str = f"{delta:+.3f}"
            if "0.000" in delta_str:
                delta_str = delta_str.replace("+", "").replace("-", "") # Handle -0.000
            
            # Format CI
            ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
            
            # Format q
            if q < 0.001:
                q_str = "$<$ 0.001"
            else:
                q_str = f"{q:.3f}"
                
            if q < 0.05:
                delta_str = f"\\textbf{{{delta_str}}}"
            
            cohort_str = f"\\texttt{{{cohort}}}" if j == 0 else ""
            comp_str = f"\\texttt{{{a}}} - \\texttt{{{b}}}"
            
            # Add some padding to match user's example indentation
            latex_row = f"				{cohort_str:13} & {comp_str:40} & {delta_str:15} & {ci_str:20} & {q_str} \\\\"
            latex_rows.append(latex_row)
            
        if i < len(cohorts) - 1:
            latex_rows.append("				\\midrule")
            
    return header + "\n" + "\n".join(latex_rows) + "\n" + footer

def generate_dataset_effect_latex(csv_path, cohorts):
    """
    Parses dataset enrichment effect CSV and generates LaTeX code for a summary table.
    """
    encoders = ["Prost40M", "UNI", "Virchow2", "H-optimus-0"]

    header = r"""\begin{table}[ht]
		\centering
		\caption{\textbf{Effect of training data enrichment on model performance.} The table shows the change in concordance index ($\Delta$) between models trained on the combined \texttt{RUMC+TCGA} splits and those trained on \texttt{RUMC-only} splits. For each test cohort and encoder, $\Delta$ values are reported alongside 95\% bootstrap CI, the exact two-sided $p$ values, and FDR-adjusted $q$ values. Positive $\Delta$ values indicate higher performance after training data enrichment, whereas negative values indicate lower performance. Values that remain significant after FDR correction ($q < 0.05$) are shown in bold.}
		\begin{tabular}{llcccc}
			\toprule
			\textbf{Cohort} & \textbf{Encoder} & $\Delta$ & \textbf{95\% CI }& $p$ & $q$ \\
			\midrule"""

    footer = r"""			\bottomrule
		\end{tabular}
		\label{table:dataset-enrichement-statistics}
	\end{table}"""

    if not os.path.exists(csv_path):
        print(f"% Warning: {csv_path} does not exist.")
        return ""

    df = pd.read_csv(csv_path)
    latex_rows = []

    for i, cohort in enumerate(cohorts):
        cohort_df = df[df['cohort'] == cohort]
        
        for j, encoder in enumerate(encoders):
            row = cohort_df[cohort_df['encoder'] == encoder]
            if row.empty:
                print(f"% Warning: encoder {encoder} not found for cohort {cohort}")
                continue
            
            row = row.iloc[0]
            delta = row['delta']
            ci_low = row['ci_low']
            ci_high = row['ci_high']
            p = row['p']
            q = row['q']

            # Format delta
            delta_str = f"{delta:+.3f}"
            if "0.000" in delta_str:
                delta_str = delta_str.replace("+", "").replace("-", "")
            
            # Format p and q
            p_str = "$<$ 0.001" if p < 0.001 else f"{p:.3f}"
            q_str = "$<$ 0.001" if q < 0.001 else f"{q:.3f}"

            if q < 0.05:
                delta_str = f"\\textbf{{{delta_str}}}"

            cohort_str = f"\\texttt{{{cohort}}}" if j == 0 else ""
            encoder_str = f"\\texttt{{{encoder}}}"

            latex_row = f"			{cohort_str:13} & {encoder_str:20} & {delta_str:15} & [{ci_low:.3f}, {ci_high:.3f}] & {p_str:10} & {q_str} \\\\"
            latex_rows.append(latex_row)
            
        if i < len(cohorts) - 1:
            latex_rows.append("			\\midrule")

    return header + "\n" + "\n".join(latex_rows) + "\n" + footer

def generate_summary_latex(summary_csv, cohorts, train_setting="rumc"):
    """
    Parses summary results CSV and generates LaTeX code for the ensemble and joint modeling table.
    """
    encoders = ["Prost40M", "UNI", "Virchow2", "H-optimus-0"]
    
    label = train_setting.upper()
    if "+" not in label:
        label += "-only"

    header = r"""\begin{table}[htbp]
		\centering
		\caption{\textbf{Ensemble and joint modeling performance of models trained on \texttt{""" + label + r"""}.} Summary of the prognostic performance of deep learning models evaluated on the four held-out test cohorts. For each cohort, the \emph{ensemble} c-index reflects the standalone performance of the deep learning model and is computed by averaging patient-level risk scores from the five \texttt{""" + label + r"""} cross-validation folds. The \emph{joint} c-index is obtained by fitting a multivariable CoxPH model using both the ensemble risk score and the CAPRA-S score as covariates. For joint models, the hazard ratio (HR) with $95$\% confidence interval (CI) and the Wald test $p$-value indicate the independent contribution of the ensemble risk score after adjusting for CAPRA-S. Bold values indicate the highest c-index observed for each cohort.}
		\resizebox{\textwidth}{!}{
			\begin{tabular}{llccccc}
				\toprule
				\textbf{""" + ("Cohort" if "+" in train_setting else "Test set") + r"""} & \textbf{Encoder} & \textbf{Ensemble} &  \textbf{Joint} & \textbf{HR (95\% CI)} & \textbf{p-value} \\
				\midrule"""

    footer = r"""				\bottomrule
			\end{tabular}
			\label{table:multivariable-cox-models-""" + train_setting.lower().replace("+", "-") + r"""}
		}
	\end{table}"""

    if not os.path.exists(summary_csv):
        print(f"% Warning: {summary_csv} does not exist.")
        return ""

    df = pd.read_csv(summary_csv)
    df = df[df['training_set'].str.lower() == train_setting.lower()]
    
    latex_rows = []

    for i, cohort in enumerate(cohorts):
        cohort_df = df[df['test_set'] == cohort]
        if cohort_df.empty:
            print(f"% Warning: no data for cohort {cohort}")
            continue

        # Find max values for bolding
        max_ens = cohort_df['ens_c_index'].max()
        max_joint = cohort_df['combined_c_index_ens'].max()

        for j, encoder in enumerate(encoders):
            row = cohort_df[cohort_df['encoder'] == encoder]
            if row.empty:
                print(f"% Warning: encoder {encoder} not found for cohort {cohort}")
                continue
            
            row = row.iloc[0]
            ens_c = row['ens_c_index']
            joint_c = row['combined_c_index_ens']
            hr_ci = row['ens_hr_95_ci']
            # Normalize dash in HR CI
            hr_ci = hr_ci.replace('–', '-')
            p = row['ens_p_value']

            # Format values
            ens_str = f"{ens_c:.3f}"
            if ens_c == max_ens:
                ens_str = f"\\textbf{{{ens_str}}}"
            
            joint_str = f"{joint_c:.3f}"
            if joint_c == max_joint:
                joint_str = f"\\textbf{{{joint_str}}}"

            p_str = "$<$ 0.001" if p < 0.001 else f"{p:.3f}"

            # Multirow for the first encoder
            if j == 0:
                cohort_str = f"\\multirow{{{len(encoders)}}}{{*}}{{\\texttt{{{cohort}}}}}"
            else:
                cohort_str = ""

            encoder_str = f"\\texttt{{{encoder}}}"
            
            latex_row = f"				{cohort_str:30} & {encoder_str:20} & {ens_str:15} & {joint_str:15} & {hr_ci:20} & {p_str} \\\\"
            latex_rows.append(latex_row)
            
        if i < len(cohorts) - 1:
            latex_rows.append("				\\midrule")

    return header + "\n" + "\n".join(latex_rows) + "\n" + footer

if __name__ == "__main__":
    pairwise_dir = "results/pairwise-tests"
    dataset_effect_csv = "results/dataset-effect.csv"
    summary_csv = "results/summary.csv"
    cohorts = ["RUMC", "PLCO", "IMP", "UHC"]
    
    print("% ======================================================================")
    print("% Summary Tables (Ensemble & Joint)")
    print("% ======================================================================")
    for train_setting in ["RUMC", "RUMC+TCGA"]:
        print(f"% --- Table for {train_setting} ---")
        summary_latex = generate_summary_latex(summary_csv, cohorts, train_setting=train_setting)
        print(summary_latex)
        print("\n")

    print("% ======================================================================")
    print("% Pairwise Comparison Tables")
    print("% ======================================================================")
    for train_setting in ["RUMC", "RUMC+TCGA"]:
        print(f"% --- Table for {train_setting} ---")
        latex_code = generate_pairwise_latex(pairwise_dir, cohorts, train_setting=train_setting)
        print(latex_code)
        print("\n")

    print("% ======================================================================")
    print("% Dataset Enrichment Effect Table")
    print("% ======================================================================")
    dataset_effect_latex = generate_dataset_effect_latex(dataset_effect_csv, cohorts)
    print(dataset_effect_latex)
