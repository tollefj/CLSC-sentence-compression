#!/bin/bash
declare -A languages

# main tests
languages=(
    ["french"]="fr"
    # ["german"]="de"
    ["polish"]="pl"
    ["hungarian"]="hu"
    # ["danish"]="da"
    ["lithuanian"]="lt"
    # ["romanian"]="ro"
    ["norwegian"]="no"
    ["malay"]="ms"
    ["albanian"]="sq"
    # ["icelandic"]="is"
    ["basque"]="eu"
)
# languages=(
#     ["norwegian"]="no"
# )
# low-resource languages
# languages=(
#     ["malay"]="ms"
#     ["albanian"]="sq"
#     ["icelandic"]="is"
#     ["basque"]="eu"
# )

for lang in "${!languages[@]}"; do
    python3 download_langs.py "en" "${languages[$lang]}"
done