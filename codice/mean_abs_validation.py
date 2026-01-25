import pandas as pd
import json
import os
import numpy as np
from sklearn.metrics import mean_absolute_error

# --- CONFIGURAZIONE ---
EXCEL_PATH = 'validation_2.xlsx'
JSON_FOLDER = 'file_json'
LABELS = [
    "OVERRELIANCE_DECISION", "AUTONOMY_LOW_SELF", "EMOTIONAL_DEPENDENCE",
    "PARASOCIAL_REL", "ANTHROPOMORPHISM_STRONG", "ADDICTIVE_USE",
    "SOCIAL_WITHDRAWAL", "SKILL_ATROPHY", "ANXIETY_ESCALATION", "CRISIS_RISK"
]

def load_json_data(folder):
    all_data = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Se il json è una lista di dizionari, li aggiungiamo
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                except Exception as e:
                    print(f"Errore nella lettura di {filename}: {e}")
    return pd.DataFrame(all_data)

def calculate_metrics(df_merged, group_name):
    metrics = {}
    for label in LABELS:
        # Rimuoviamo eventuali NaN per il calcolo della distanza sulla singola label
        valid_subset = df_merged[[f"{label}_xlsx", f"{label}_json"]].dropna()
        if not valid_subset.empty:
            mae = mean_absolute_error(valid_subset[f"{label}_xlsx"], valid_subset[f"{label}_json"])
            metrics[label] = round(mae, 4)
        else:
            metrics[label] = None
    
    # Distanza media globale per questo gruppo
    all_values = list(filter(None, metrics.values()))
    metrics['GLOBAL_AVERAGE_DISTANCE'] = round(np.mean(all_values), 4) if all_values else 0
    return metrics

# 1. Caricamento dati Excel
print("Caricamento file Excel...")
df_xlsx = pd.read_excel(EXCEL_PATH)
# Assicuriamoci che gli ID siano stringhe per il matching
df_xlsx['conversation_id'] = df_xlsx['conversation_id'].astype(str)
df_xlsx['message_id'] = df_xlsx['message_id'].astype(str)

# 2. Caricamento dati JSON
print("Caricamento file JSON...")
df_json = load_json_data(JSON_FOLDER)
df_json['conversation_id'] = df_json['conversation_id'].astype(str)
df_json['message_id'] = df_json['message_id'].astype(str)

# 3. Merge dei dataset su conversation_id e message_id
# Usiamo suffissi per distinguere le colonne
df_merged = pd.merge(
    df_xlsx, 
    df_json, 
    on=['conversation_id', 'message_id'], 
    how='inner', 
    suffixes=('_xlsx', '_json')
)

if df_merged.empty:
    print("Nessun match trovato tra Excel e JSON. Controlla gli ID.")
    exit()

# 4. Calcolo Distanza per Singola Istanza
print("Calcolo distanze per istanza...")
# Calcoliamo la distanza media tra le 10 label per ogni riga
df_merged['instance_distance'] = df_merged.apply(
    lambda row: np.mean([abs(row[f"{l}_xlsx"] - row[f"{l}_json"]) for l in LABELS]), axis=1
)

# 5. Analisi per Tipo di Dataset
print("\n--- REPORT PER DATASET ---")
dataset_report = []
for dataset_name, group in df_merged.groupby('name_original_dataset'):
    m = calculate_metrics(group, dataset_name)
    m['dataset'] = dataset_name
    dataset_report.append(m)

df_dataset_report = pd.DataFrame(dataset_report).set_index('dataset')
print(df_dataset_report)

# 6. Analisi Globale
print("\n--- REPORT GLOBALE ---")
global_metrics = calculate_metrics(df_merged, "Global")
for k, v in global_metrics.items():
    print(f"{k}: {v}")

# 7. Salvataggio Risultati
df_merged[['conversation_id', 'message_id', 'instance_distance']].to_csv('distanze_istanze.csv', index=False)
df_dataset_report.to_csv('report_distanze_dataset.csv')
print("\nFile 'distanze_istanze.csv' e 'report_distanze_dataset.csv' generati con successo.")