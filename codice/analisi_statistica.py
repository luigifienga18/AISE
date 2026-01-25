import pandas as pd

def analizza_dataset_completo(file_path):
    target_datasets = ['PsyDial-D4.json', 'Conv-GPT', 'Conv-Haiku']
    
    try:
        # Caricamento del dataset completo con codifica utf-8-sig
        df_full = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # --- 1. ANALISI TESTO (Su TUTTI i messaggi del file) ---
        # Calcolo del numero di parole per ogni messaggio
        df_full['word_count'] = df_full['text'].fillna('').apply(lambda x: len(str(x).split()))
        
        mean_words_all = df_full['word_count'].mean()
        max_words_all = df_full['word_count'].max()
        above_mean_words_all = (df_full['word_count'] > mean_words_all).sum()
        around_max_words_all = (df_full['word_count'] >= max_words_all * 0.9).sum()
        
        # Conteggio specifico richiesto: messaggi > 512 parole
        over_512_count = (df_full['word_count'] > 512).sum()
        over_512_percent = (over_512_count / len(df_full)) * 100
        
        # --- 2. ANALISI CONVERSAZIONI (Solo sui dataset specificati) ---
        df_sub = df_full[df_full['name_original_dataset'].isin(target_datasets)].copy()
        
        stats_conv = None
        if not df_sub.empty:
            conv_counts = df_sub.groupby('conversation_id')['message_id'].count()
            mean_conv = conv_counts.mean()
            max_conv = conv_counts.max()
            above_mean_conv = (conv_counts > mean_conv).sum()
            around_max_conv = (conv_counts >= max_conv * 0.9).sum()
            stats_conv = (mean_conv, max_conv, above_mean_conv, around_max_conv, len(conv_counts))

        # --- OUTPUT RISULTATI ---
        print("="*65)
        print("ANALISI TESTO: GLOBALE (Tutti i messaggi nel dataset)")
        print("="*65)
        print(f"Messaggi totali:           {len(df_full)}")
        print(f"Media parole:              {mean_words_all:.2f}")
        print(f"Massimo numero parole:     {max_words_all}")
        print(f"Messaggi > media:          {above_mean_words_all}")
        print(f"Messaggi > 512 parole:     {over_512_count} ({over_512_percent:.2f}%)")
        print(f"Messaggi vicini al max:    {around_max_words_all}")
        
        print("\n" + "="*65)
        print(f"ANALISI CONVERSAZIONI: SOLO SUBSET")
        print(f"Fonti: {', '.join(target_datasets)}")
        print("="*65)
        if stats_conv:
            m_c, mx_c, ab_c, ar_c, tot_c = stats_conv
            print(f"Conversazioni totali:      {tot_c}")
            print(f"Media messaggi/conv:       {m_c:.2f}")
            print(f"Massimo numero messaggi:   {mx_c}")
            print(f"Conversazioni > media:     {ab_c}")
            print(f"Conversazioni vicine al max: {ar_c}")
        else:
            print("Nessun dato trovato per i dataset specificati.")
        print("="*65)

    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    analizza_dataset_completo('nuovo_dataset_5000.csv')