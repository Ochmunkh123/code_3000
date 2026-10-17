import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = ["age", "gender", "zip3"]

    # Build every possible exact match. An anonymized record is considered
    # re-identified only when its quasi-identifiers point to exactly one row
    # in the auxiliary data.
    candidates = anon_df.merge(
        aux_df,
        on=quasi_identifiers,
        how="inner",
    )
    candidate_counts = candidates.groupby("anon_id")["anon_id"].transform("size")

    unique_matches = candidates.loc[
        candidate_counts.eq(1), ["anon_id", "name"]
    ].copy()
    return unique_matches.rename(columns={"name": "matched_name"}).reset_index(
        drop=True
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    return len(matches_df) / len(anon_df)
