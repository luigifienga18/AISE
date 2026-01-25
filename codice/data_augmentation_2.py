import pandas as pd
import uuid

def custom_oversampling_v5(input_file, output_file):
    print(f"--- Inizio Elaborazione (Regole V5) ---")
    
    # 1. Caricamento
    try:
        df = pd.read_csv(input_file)
        print(f"Dataset caricato: {len(df)} righe totali.")
    except FileNotFoundError:
        print("ERRORE: File non trovato.")
        return

    # --- FASE 0: PROTEZIONE ASSISTANT ---
    if 'user_role' not in df.columns:
        print("ERRORE: Manca la colonna 'user_role'.")
        return

    df_assistants = df[df['user_role'] == 'assistant']
    df_users = df[df['user_role'] != 'assistant'].copy()

    # Definiamo le colonne Label
    label_cols = [
        'OVERRELIANCE_DECISION', 'AUTONOMY_LOW_SELF', 'EMOTIONAL_DEPENDENCE',
        'PARASOCIAL_REL', 'ANTHROPOMORPHISM_STRONG', 'ADDICTIVE_USE',
        'SOCIAL_WITHDRAWAL', 'SKILL_ATROPHY', 'ANXIETY_ESCALATION', 'CRISIS_RISK'
    ]
    
    # --- FASE 1: UNDERSAMPLING ZERI ---
    # Identifichiamo le righe con somma 0
    df_users['is_all_zeros'] = df_users[label_cols].sum(axis=1) == 0
    
    # Separiamo Zeri e Non-Zeri
    df_zeros = df_users[df_users['is_all_zeros'] == True]
    df_non_zeros = df_users[df_users['is_all_zeros'] == False]
    
    # Teniamo il 40% degli zeri
    df_zeros_kept = df_zeros.sample(frac=1, random_state=42)
    
    print(f"\n[Undersampling Zeri]")
    print(f"Originali: {len(df_zeros)} -> Mantenuti: {len(df_zeros_kept)}")

    # --- FASE 2: PREPARAZIONE OVERSAMPLING ---
    candidates = df_non_zeros.copy()

    # A. FILTRO ESCLUSIONE
    # Escludiamo chi ha 1 in Autonomy o Anxiety
    mask_exclude = (candidates['AUTONOMY_LOW_SELF'] == 1) | (candidates['ANXIETY_ESCALATION'] == 1)
    valid_candidates = candidates[~mask_exclude].copy()
    
    print(f"\n[Filtro Esclusione]")
    print(f"User con label attive: {len(candidates)}")
    print(f"Esclusi (Autonomy=1 o Anxiety=1): {mask_exclude.sum()}")
    print(f"Candidati validi per oversampling: {len(valid_candidates)}")

    # B. CONTEGGI
    # Contiamo quanti 2 ci sono
    valid_candidates['count_twos'] = valid_candidates[label_cols].eq(2).sum(axis=1)
    # Contiamo quanti 1 ci sono (su tutte le colonne, tanto quelle escluse sono sicuramente != 1 qui)
    valid_candidates['count_ones'] = valid_candidates[label_cols].eq(1).sum(axis=1)

    # --- FASE 3: SELEZIONE IN BASE ALLE REGOLE ---
    
    # Regola 1: >= 3 valori '2' -> 100%
    mask_3_twos = valid_candidates['count_twos'] >= 3
    sample_3_twos = valid_candidates[mask_3_twos].sample(frac=1.0, random_state=42)
    
    # Regola 2: Esattamente 2 valori '2' -> 75%
    mask_2_twos = valid_candidates['count_twos'] == 2
    sample_2_twos = valid_candidates[mask_2_twos].sample(frac=0.75, random_state=42)
    
    # Regola 3: Esattamente 1 valore '2' -> 30%
    mask_1_two = valid_candidates['count_twos'] == 1
    sample_1_two = valid_candidates[mask_1_two].sample(frac=0.30, random_state=42)
    
    # Regola 4: Più di 4 label a '1' (quindi >= 5), ma SOLO se non hanno '2'
    # (Se avessero dei 2, sarebbero già stati gestiti sopra. Questo evita duplicazioni multiple della stessa riga)
    mask_high_ones = (valid_candidates['count_twos'] == 0) & (valid_candidates['count_ones'] >= 4)
    sample_high_ones = valid_candidates[mask_high_ones].sample(frac=1, random_state=42)

    # Report Selezione
    print(f"\n[Selezione Oversampling]")
    print(f"1. Gruppo '3+ Twos' (100%): {len(sample_3_twos)} presi su {mask_3_twos.sum()}")
    print(f"2. Gruppo '2 Twos'  (75%):  {len(sample_2_twos)} presi su {mask_2_twos.sum()}")
    print(f"3. Gruppo '1 Two'   (30%):  {len(sample_1_two)} presi su {mask_1_two.sum()}")
    print(f"4. Gruppo '>3 Ones' (60%):  {len(sample_high_ones)} presi su {mask_high_ones.sum()} (tra quelli senza 2)")

    # Unione dei campioni
    df_to_oversample = pd.concat([sample_3_twos, sample_2_twos, sample_1_two, sample_high_ones])
    print(f"-> TOTALE NUOVE ISTANZE: {len(df_to_oversample)}")

    # --- FASE 4: APPLICAZIONE MODIFICHE ---
    if len(df_to_oversample) > 0:
        # Modifica Conversation ID
        df_to_oversample['conversation_id'] = (
            df_to_oversample['conversation_id'].astype(str) + 
            '_aug_' + 
            [str(uuid.uuid4())[:5] for _ in range(len(df_to_oversample))]
        )

        # Modifica Dataset Name
        suffix = '_oversampled'
        if 'name_original_dataset' in df_to_oversample.columns:
            df_to_oversample['name_original_dataset'] = df_to_oversample['name_original_dataset'].astype(str) + suffix
        else:
            df_to_oversample['name_original_dataset'] = 'unknown' + suffix
    else:
        df_to_oversample = pd.DataFrame()

    # --- FASE 5: UNIONE FINALE ---
    df_final = pd.concat([
        df_zeros_kept,      # Zeri ridotti
        df_non_zeros,       # Originali User (TUTTI, anche quelli non selezionati per la copia)
        df_to_oversample,   # Copie create
        df_assistants       # Assistant originali
    ], axis=0)

    # Pulizia
    cols_drop = ['is_all_zeros', 'count_twos', 'count_ones']
    for c in cols_drop:
        if c in df_final.columns:
            df_final.drop(columns=[c], inplace=True)
            
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Salvataggio
    df_final.to_csv(output_file, index=False)
    print(f"\n--- Riepilogo Finale ---")
    print(f"Righe totali: {len(df_final)}")
    print(f"File salvato in: {output_file}")

# --- ESECUZIONE ---
if __name__ == "__main__":
    # INSERISCI QUI I PERCORSI CORRETTI
    input_csv = r"C:\Users\matte\Dropbox\Il mio PC (LAPTOP-EMJU3OME)\Desktop\università\MAGISTRALE\AISE\PROGETTO\dataset_bilanciato_multi.csv"
    output_csv = ".\dataset_oversampled_pesante.csv"
    
    custom_oversampling_v5(input_csv, output_csv)