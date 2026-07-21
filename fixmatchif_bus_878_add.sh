for ifrank_w in 1.0;
do
    for corrT in 0.01;
    do
        for if_lambda in 10; 
        do
            for i in 1;
            do
                seed=$RANDOM
                tmp_yaml="temp_fixmatchif_bus_878_add_ifw${if_lambda}_corrT${corrT}_$i.yaml"
                sed "s/save_name:.*/save_name: fixmatchif_bus_878_add_ifw${if_lambda}_corrT${corrT}_$i/" config/usb_cv/fixmatch/fixmatchif_bus_878_multiply_1.yaml | \
                sed "s/seed:.*/seed: $seed/" | \
                sed "s/resume:.*/resume: False/" | \
                sed "s/corrT:.*/corrT: $corrT/" | \
                sed "s/if_lambda:.*/if_lambda: $if_lambda/" | \
                sed "s/ifrank_loss_weight:.*/ifrank_loss_weight: $ifrank_w/" | \
                sed "s/combine:.*/combine: 'add'/" > "$tmp_yaml"

                python3 train.py --c "$tmp_yaml"
                rm "$tmp_yaml"
            done
        done
    done
done
