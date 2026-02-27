"""
linguistic_redlining.py
-----------------------
Extracts explicit examples of "Linguistic Redlining" from the LLM-scored corpus.
Linguistic Redlining occurs when an AI system penalizes marginalized groups for
their use of dialect, cultural phrasing, or AAVE under the guise of "poor grammar",
even when their argumentation and coherence are brilliant (recognized by human experts).

This script:
1. Identifies marginalized students (e.g., Black/African American or Hispanic)
   who wrote excellent essays (Human Score = 5 or 6).
2. Finds cases where the AI gave them terrible scores (AI Score = 1, 2, or 3).
3. Breaks down the sub-dimensional scores to prove the redlining:
   Does the AI concede their Argumentation (C1=4 or 5) but nuke their score
   because of Grammar (C4=1 or 2)?
4. Dumps the essays to a markdown file for the white paper.

Author: Sai Chaitanya Pachipulusu (Independent Psychometric Audit, 2026)
"""

import pandas as pd
from pathlib import Path

def extract_redlining_examples(scored_corpus_path: str = "data/scored_corpus.csv", out_dir: str = "reports/"):
    df = pd.read_csv(scored_corpus_path)
    
    # Needs demographics
    if "race_ethnicity" not in df.columns:
        print("No race_ethnicity column found. Cannot extract linguistic redlining examples.")
        return

    # Filter for marginalized groups writing excellent essays
    marginalized = ["Black/African American", "Hispanic/Latino"]
    excellent_human = df[(df["human_score"] >= 5) & (df["race_ethnicity"].isin(marginalized))]

    # Find where AI heavily penalized them
    redlined = excellent_human[excellent_human["ai_score"] <= 3].copy()
    
    # Calculate the Grammar Penalty (Difference between Argumentation and Grammar)
    # The hypothesis: AI recognizes the argument is good, but kills the score due to dialetic "Grammar"
    if "ai_c1" in redlined.columns and "ai_c4" in redlined.columns:
        redlined["grammar_penalty"] = redlined["ai_c1"] - redlined["ai_c4"]
        redlined = redlined.sort_values(by="grammar_penalty", ascending=False)
    else:
        # Fallback if subscores aren't there
        redlined["residual"] = redlined["human_score"] - redlined["ai_score"]
        redlined = redlined.sort_values(by="residual", ascending=False)

    out = Path(out_dir) / "LINGUISTIC_REDLINING_EXAMPLES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Linguistic Redlining Examples\n\n")
        f.write("These essays demonstrate highly competent communication (Human Score 5/6) ")
        f.write("by marginalized students, which were severely penalized by the AI (Score <= 3).\n\n")
        
        f.write(f"**Total redlined essays found:** {len(redlined)}\n\n")
        
        for i, (_, row) in enumerate(redlined.head(10).iterrows()):
            f.write(f"## Example {i+1} : {row['race_ethnicity']} Student\n")
            f.write(f"- **Human Expert Score:** {row['human_score']}/6\n")
            f.write(f"- **AI Holistic Score:**  {row['ai_score']}/6\n")
            if "ai_c1" in row:
                f.write(f"  - AI Argumentation (C1): {row['ai_c1']}/6\n")
                f.write(f"  - AI Grammar (C4):       {row['ai_c4']}/6\n")
                
            f.write("\n**Essay Text:**\n")
            f.write(f"> {row['essay_text'].strip()}\n\n")
            f.write("---\n\n")

    print(f"\n[Redlining] Extracted {len(redlined)} strong examples of Linguistic Redlining.")
    print(f"[Redlining] Saved to: {out.resolve()}")

if __name__ == "__main__":
    extract_redlining_examples()
