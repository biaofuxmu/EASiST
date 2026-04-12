MUSTC_ROOT=/path/to/MuST-C
OUTPUT_PATH=data/offline_st

for lang in de es
do
    python data_process/prep_mustc.py \
        --data-root ${MUSTC_ROOT} \
        --output-path ${OUTPUT_PATH} \
        --tgt-lang ${lang}
done
