import pandas as pd
import os

print('Current working directory:', os.getcwd())
print('File exists:', os.path.exists('data/scored_corpus.csv'))

if os.path.exists('data/scored_corpus.csv'):
    print('File size:', os.path.getsize('data/scored_corpus.csv'))
    try:
        df = pd.read_csv('data/scored_corpus.csv')
        print('Successfully read CSV')
        print('Columns:', list(df.columns))
        print('AI Score Distribution:')
        print(df['ai_score'].value_counts().sort_index())
        print('\nHuman Score Distribution:')
        print(df['human_score'].value_counts().sort_index())
        print('\nHigh AI Score (score > 4) by Group:')
        print(df.groupby('group')['ai_score'].apply(lambda x: (x > 4).sum()))
    except Exception as e:
        print('Error reading CSV:', type(e).__name__, str(e))
        import traceback
        print(traceback.format_exc())
