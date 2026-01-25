import pandas as pd
import uuid

# 2. Definizione Colonne Label
label_cols = [
        'OVERRELIANCE_DECISION', 'AUTONOMY_LOW_SELF', 'EMOTIONAL_DEPENDENCE',
        'PARASOCIAL_REL', 'ANTHROPOMORPHISM_STRONG', 'ADDICTIVE_USE',
        'SOCIAL_WITHDRAWAL', 'SKILL_ATROPHY', 'ANXIETY_ESCALATION', 'CRISIS_RISK'
    ]
dtype_dict_1 = {col: 'Int64' for col in label_cols}


def tiered_balance_dataset_protected(input_file, output_file):
    print(f"--- Inizio Elaborazione ---")
    
    # 1. Caricamento
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig', dtype=dtype_dict_1)
        print(f"Dataset caricato: {len(df)} righe totali.")
    except FileNotFoundError:
        print("ERRORE: File non trovato. Controlla il percorso.")
        return

    # --- FASE PRELIMINARE: PROTEZIONE ASSISTANT ---
    if 'user_role' not in df.columns:
        print("ERRORE: La colonna 'user_role' non esiste nel dataset.")
        return

    # Separiamo gli assistant (da proteggere) dagli altri (da bilanciare)
    df_assistants = df[df['user_role'] == 'assistant']
    df_to_process = df[df['user_role'] != 'assistant'].copy()

    print(f"\n--- Separazione Ruoli ---")
    print(f"Righe Assistant (protette, non toccate): {len(df_assistants)}")
    print(f"Righe User/Altro (soggette a bilanciamento): {len(df_to_process)}")


    # Verifica presenza colonne
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        print(f"ERRORE: Colonne label mancanti: {missing}")
        return

    # Calcolo label attive SOLO sul dataset da processare
    df_to_process['active_label_count'] = df_to_process[label_cols].gt(0).sum(axis=1)

    # --- CONFIGURAZIONE REGOLE OVERSAMPLING ---
    sampling_rules = {
        2: 0.20,
        3: 0.40,
        4: 0.65,
        5: 0.80
    }
    # Per > 6 la percentuale sarà 1.0 (100%)

    # --- FASE 1: UNDERSAMPLING ZERI (Solo User) ---
    df_zeros = df_to_process[df_to_process['active_label_count'] == 0]
    
    # Manteniamo il 40% degli zeri user
    df_zeros_kept = df_zeros.sample(frac=0.4, random_state=42)
    
    # --- FASE 2: PREPARAZIONE E OVERSAMPLING (Solo User) ---
    # Isoliamo le righe user con label > 0 (da mantenere originali)
    df_non_zeros = df_to_process[df_to_process['active_label_count'] > 0]
    
    new_rows_list = []
    report_stats = {}

    # Otteniamo i gruppi presenti
    unique_counts = sorted(df_to_process['active_label_count'].unique())

    for c in unique_counts:
        subset = df_to_process[df_to_process['active_label_count'] == c]
        orig_count = len(subset)
        added_count = 0

        # Logica Oversampling (Solo per >= 2 label)
        if c >= 2:
            if c in sampling_rules:
                frac = sampling_rules[c]
            else:
                frac = 1.0 # 100% per label > 6

            # Estrazione copie
            samples_to_add = subset.sample(frac=frac, random_state=42)
            added_count = len(samples_to_add)

            if added_count > 0:
                new_rows_list.append(samples_to_add)
        
        report_stats[c] = {'original': orig_count, 'added': added_count}

    # --- FASE 3: MODIFICA DELLE COPIE ---
    if new_rows_list:
        df_oversampled = pd.concat(new_rows_list)
        
        # Modifica ID univoco
        df_oversampled['conversation_id'] = (
            df_oversampled['conversation_id'].astype(str) + 
            '_aug_' + 
            [str(uuid.uuid4())[:5] for _ in range(len(df_oversampled))]
        )

        # Modifica nome dataset
        suffix = '_oversampled'
        if 'name_original_dataset' in df_oversampled.columns:
            df_oversampled['name_original_dataset'] = df_oversampled['name_original_dataset'].astype(str) + suffix
        else:
            df_oversampled['name_original_dataset'] = 'unknown' + suffix
    else:
        df_oversampled = pd.DataFrame()

    # --- FASE 4: CREAZIONE DATASET FINALE ---
    # Uniamo:
    # 1. User Zeri Ridotti
    # 2. User Non-Zeri Originali
    # 3. User Oversampled (Copie)
    # 4. ASSISTANT ORIGINALI (Protetti)
    
    df_final = pd.concat([df_zeros_kept, df_non_zeros, df_oversampled, df_assistants], axis=0)
    
    # Rimuoviamo la colonna temporanea (se presente anche negli assistant per errore di allineamento, pandas la gestisce)
    if 'active_label_count' in df_final.columns:
        df_final.drop(columns=['active_label_count'], inplace=True)
    
    # Shuffle finale per mischiare user e assistant
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- FASE 5: REPORT ---
    print("\n" + "="*60)
    print(f"{'GRUPPO (SOLO USER)':<20} | {'ORIGINALI':<12} | {'AGGIUNTE (COPIE)':<15}")
    print("-" * 60)
    
    for c in sorted(report_stats.keys()):
        orig = report_stats[c]['original']
        added = report_stats[c]['added']
        
        if c == 0:
            note = f"(Ridotte a {len(df_zeros_kept)})"
            print(f"{c} Label (Zero)        | {orig:<12} | 0 {note}")
        elif c == 1:
            print(f"{c} Label (Single)      | {orig:<12} | {added:<15}")
        else:
            print(f"{c} Label               | {orig:<12} | {added:<15}")

    print("-" * 60)
    print(f"Righe Assistant (Invariate): {len(df_assistants)}")
    print(f"Righe User Totali (Dopo bilanciamento): {len(df_final) - len(df_assistants)}")
    print("-" * 60)
    print(f"{'TOTALE DATASET':<20} | {len(df_final):<12} |")
    print("="*60 + "\n")

    # Salvataggio
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"File salvato correttamente in:\n{output_file}")


# ESEMPIO DI UTILIZZO
if __name__ == "__main__":
    # Ricordati la 'r' prima del percorso se sei su Windows e usi backslash
    input_csv = r"C:\Users\matte\Dropbox\Il mio PC (LAPTOP-EMJU3OME)\Desktop\università\MAGISTRALE\AISE\PROGETTO\prima_annotazione\dataset_ridotto_aggiornato.csv" 
    output_csv = ".\dataset_bilanciato_multi.csv"
    
    tiered_balance_dataset_protected(input_csv, output_csv)
   