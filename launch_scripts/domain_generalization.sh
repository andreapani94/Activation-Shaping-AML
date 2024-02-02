#!/bin/bash

declare -a domains=("art_painting" "cartoon" "sketch" "photo")
declare -a source_domains=()

target_domain=${1}

# excluding the target domain from the domains
for domain in "${domains[@]}"; 
do
    if [ "$domain" != "$target_domain" ];
    then
        source_domains+=("$domain")
    fi
done

python main.py \
--experiment=domain_generalization \
--experiment_name=domain_generalization/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domains': '${source_domains[*]}', 'target_domain': '${target_domain}' }" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 \

