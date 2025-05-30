import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

loso_results_string = """
Fold 1 (Test Subj: VP001-EEG): Acc=39.11%, F1=35.57%, AUC=0.5752
Fold 2 (Test Subj: VP002-EEG): Acc=32.68%, F1=20.53%, AUC=0.4916
Fold 3 (Test Subj: VP003-EEG): Acc=38.76%, F1=32.72%, AUC=0.5440
Fold 4 (Test Subj: VP004-EEG): Acc=34.91%, F1=24.75%, AUC=0.4732
Fold 5 (Test Subj: VP005-EEG): Acc=46.68%, F1=37.07%, AUC=0.8320
Fold 6 (Test Subj: VP006-EEG): Acc=39.70%, F1=29.52%, AUC=0.5453
Fold 7 (Test Subj: VP007-EEG): Acc=41.16%, F1=39.90%, AUC=0.6853
Fold 8 (Test Subj: VP008-EEG): Acc=41.56%, F1=40.99%, AUC=0.6379
Fold 9 (Test Subj: VP009-EEG): Acc=25.28%, F1=15.65%, AUC=0.4523
Fold 10 (Test Subj: VP010-EEG): Acc=32.84%, F1=26.92%, AUC=0.5603
Fold 11 (Test Subj: VP011-EEG): Acc=46.54%, F1=43.45%, AUC=0.6672
Fold 12 (Test Subj: VP012-EEG): Acc=39.13%, F1=31.02%, AUC=0.6411
Fold 13 (Test Subj: VP013-EEG): Acc=44.10%, F1=34.28%, AUC=0.6856
Fold 14 (Test Subj: VP014-EEG): Acc=44.54%, F1=39.63%, AUC=0.7422
Fold 15 (Test Subj: VP015-EEG): Acc=40.28%, F1=33.92%, AUC=0.5554
Fold 16 (Test Subj: VP016-EEG): Acc=40.44%, F1=31.74%, AUC=0.5992
Fold 17 (Test Subj: VP017-EEG): Acc=60.85%, F1=49.84%, AUC=0.8001
Fold 18 (Test Subj: VP018-EEG): Acc=49.20%, F1=48.58%, AUC=0.6787
Fold 19 (Test Subj: VP019-EEG): Acc=36.37%, F1=31.99%, AUC=0.7799
Fold 20 (Test Subj: VP020-EEG): Acc=51.19%, F1=48.82%, AUC=0.6995
Fold 21 (Test Subj: VP021-EEG): Acc=43.66%, F1=37.75%, AUC=0.6856
Fold 22 (Test Subj: VP022-EEG): Acc=49.61%, F1=46.31%, AUC=0.7351
Fold 23 (Test Subj: VP023-EEG): Acc=54.29%, F1=47.66%, AUC=0.7064
Fold 24 (Test Subj: VP024-EEG): Acc=41.53%, F1=28.55%, AUC=0.5941
Fold 25 (Test Subj: VP025-EEG): Acc=39.08%, F1=32.85%, AUC=0.5209
Fold 26 (Test Subj: VP026-EEG): Acc=36.60%, F1=30.03%, AUC=0.6429
"""
# ==============================================================================

def parse_results_from_string(results_string):
    """
    Parses LOSO CV results from a multiline string.
    """
    import re 
    results_data = []
    for line in results_string.strip().split('\n'):
        line = line.strip()
        if line.startswith("Fold"):
            match = re.search(r"Fold (\d+) \(Test Subj: (VP\d{3}-EEG)\): Acc=([\d.]+)%, F1=([\d.]+)%, AUC=([\d.]+)", line)
            if match:
                results_data.append({
                    "Fold": int(match.group(1)),
                    "Test Subject": match.group(2),
                    "Accuracy (%)": float(match.group(3)),
                    "F1-macro (%)": float(match.group(4)),
                    "AUC": float(match.group(5))
                })
    return pd.DataFrame(results_data)

def plot_loso_summary(df_results, title="LOSO Cross-Validation Test Metrics per Subject", save_filename=None):
    """
    Plots a summary bar chart of test metrics per fold from a DataFrame.
    """
    if df_results.empty:
        print("DataFrame is empty. No data to plot.")
        return

    num_folds = len(df_results)

    fold_labels = [f"S{row['Fold']}\n({row['Test Subject']})" for index, row in df_results.iterrows()]


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9)) 

    bar_width = 0.25
    index = np.arange(num_folds)

    bar_acc = ax.bar(index - bar_width, df_results["Accuracy (%)"], bar_width, label='Accuracy (%)', color='deepskyblue')
    bar_f1 = ax.bar(index, df_results["F1-macro (%)"], bar_width, label='F1-macro (%)', color='salmon')
    bar_auc = ax.bar(index + bar_width, df_results["AUC"] * 100, bar_width, label='AUC (*100)', color='lightgreen')

    ax.set_xlabel("Test Subject (Fold Left Out)", fontsize=14)
    ax.set_ylabel("Metric Value (%)", fontsize=14)
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xticks(index)
    ax.set_xticklabels(fold_labels, rotation=75, ha="right", fontsize=10) 
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1,1)) #
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 105)

   
    def add_value_labels(bars, decimal_places=1):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.{decimal_places}f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=90) 

    add_value_labels(bar_acc)
    add_value_labels(bar_f1)
    add_value_labels(bar_auc)
    
    # Adicionar linhas de média
    avg_acc_val = df_results["Accuracy (%)"].mean()
    avg_f1_val = df_results["F1-macro (%)"].mean()
    avg_auc_val = df_results["AUC"].mean() * 100

    ax.axhline(avg_acc_val, color='dodgerblue', linestyle='--', linewidth=1.5, label=f'Avg Acc: {avg_acc_val:.2f}%')
    ax.axhline(avg_f1_val, color='darkred', linestyle='--', linewidth=1.5, label=f'Avg F1: {avg_f1_val:.2f}%')
    ax.axhline(avg_auc_val, color='darkgreen', linestyle='--', linewidth=1.5, label=f'Avg AUC: {avg_auc_val/100:.4f}')
    
    # Atualizar legenda para incluir as médias
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))


    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para a legenda externa
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        print(f"Summary plot saved to {save_filename}")
    plt.show()

# --- Execução Principal do Script ---
if __name__ == "__main__":
    # 1. Parse os resultados da string
    results_df = parse_results_from_string(loso_results_string)

    if not results_df.empty:
        # 2. Exibir a tabela de resultados
        print("--- LOSO CV Test Results Table ---")
        # Para melhor formatação da tabela no console:
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        print(results_df)
        print("-" * 30)

        # 3. Calcular e imprimir métricas médias (opcional, já que o gráfico mostrará)
        avg_acc = results_df["Accuracy (%)"].mean()
        std_acc = results_df["Accuracy (%)"].std()
        avg_f1 = results_df["F1-macro (%)"].mean()
        std_f1 = results_df["F1-macro (%)"].std()
        avg_auc = results_df["AUC"].mean()
        std_auc = results_df["AUC"].std()

        print("\n--- Average Test Metrics (Calculated by this script) ---")
        print(f"Average Accuracy: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
        print(f"Average F1-macro: {avg_f1:.2f}% (+/- {std_f1:.2f}%)")
        print(f"Average AUC:        {avg_auc:.4f} (+/- {std_auc:.4f})")

        # 4. Gerar e mostrar/salvar o gráfico resumo
        plot_loso_summary(results_df, save_filename="loso_cv_summary_chart.png")
    else:
        print("No results could be parsed from the input string.")