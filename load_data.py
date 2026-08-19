from common import basedir
from cytof.data import (
    load_cytof_from_synapse,
    load_ids_from_uniprot,
    load_proteomics_from_synapse,
    load_transcriptomics_from_synapse,
    process_petab_cytof,
    process_petab_proteomics,
    process_petab_transcriptomics,
)

if __name__ == "__main__":
    measurement_table_cytof, id_vars = load_cytof_from_synapse()
    measurement_table_cytof = process_petab_cytof(
        measurement_table_cytof, id_vars
    )
    (basedir / "data").mkdir(exist_ok=True)
    measurement_table_cytof.to_csv(basedir / "data" / "cytof.csv")
    measurement_table_proteomics = load_proteomics_from_synapse()
    up_ids = load_ids_from_uniprot(
        measurement_table_proteomics["UPID"].unique()
    )
    # missing gene names: A2VCL2, A8MUA0, O00370, Q6ZSR9
    # A2VCL2: dropped from uniprot, CCDC162P https://varsome.com/gene/hg19/CCDC162P
    # A8MUA0: Putative UPF0607 protein, SPATA6: https://varsome.com/gene/hg19/SPATA6
    # O00370: ORF2p: https://www.uniprot.org/uniprotkb/O00370/entry
    # Q6ZSR9: Uncharacterized protein FLJ45252: https://www.uniprot.org/uniprotkb/Q6ZSR9/entry
    up_ids["A2VCL2"] = "CCDC162P"
    up_ids["A8MUA0"] = "SPATA6"
    up_ids["O00370"] = "ORF2P"
    up_ids["Q6ZSR9"] = "FLJ45252"

    measurement_table_proteomics.loc[
        :, "GENENAME"
    ] = measurement_table_proteomics["UPID"].apply(lambda x: up_ids.get(x))
    measurement_table_proteomics = process_petab_proteomics(
        measurement_table_proteomics
    )
    measurement_table_proteomics.drop(
        columns=["UPID", "GENENAME"], inplace=True
    )
    measurement_table_proteomics.to_csv(basedir / "data" / "proteomics.csv")

    measurement_table_transcriptomics = load_transcriptomics_from_synapse()
    measurement_table_transcriptomics = process_petab_transcriptomics(
        measurement_table_transcriptomics
    )
    measurement_table_transcriptomics.drop(columns=["GENENAME"], inplace=True)

    measurement_table_transcriptomics.to_csv(
        basedir / "data" / "transcriptomics.csv"
    )
