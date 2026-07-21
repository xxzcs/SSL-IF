for ifrank_w in 1.0;
do
    for corrT in 0.01 0.02 0.03 0.04 0.05;
    do
        for if_lambda in 10 8; 
        do
            for i in 1 2 3 4 5;
            do
                seed=$RANDOM
                tmp_yaml="temp_fixmatchif_bus_878_multiplyo_ifw${if_lambda}_corrT${corrT}_$i.yaml"
                sed "s/save_name:.*/save_name: fixmatchif_bus_878_multiplyo_ifw${if_lambda}_corrT${corrT}_$i/" config/usb_cv/fixmatch/fixmatchif_bus_878_multiply_1.yaml | \
                sed "s/seed:.*/seed: $seed/" | \
                sed "s/resume:.*/resume: False/" | \
                sed "s/corrT:.*/corrT: $corrT/" | \
                sed "s/if_lambda:.*/if_lambda: $if_lambda/" | \
                sed "s/ifrank_loss_weight:.*/ifrank_loss_weight: $ifrank_w/" | \
                sed "s/combine:.*/combine: 'multiplyo'/" > "$tmp_yaml"

                python3 train.py --c "$tmp_yaml"
                rm "$tmp_yaml"
            done
        done
    done
done
