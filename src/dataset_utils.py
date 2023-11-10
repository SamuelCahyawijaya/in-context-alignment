import os, sys
from os.path import exists
import json
import glob
import pandas as pd
import datasets

# NusaWrites Data Loading
dataset_mapping = { 
    # key: (test_dset, icl_dset, x-icl_lang, iia_dset, itc_dset, ioa_dset)
    'americasnli': ('americasnli_test_dset', 'icl_americasnli_dset', 'spa', 'americasnli_combined_dev_dset', 'americasnli_combined_dev_dset', 'americasnli_label_map.csv'),
    'masakhanews': ('masakhanews_test_dset', 'icl_masakhanews_dset', 'eng', 'mafand_rand_label_dset', 'mafand_rand_label_dset', 'masakhanews_label_map.csv'),
    'nusatranslation': ('nt_senti_test_dset', 'icl_nusax_senti_dset', 'eng', 'nusax_combined_eng_dset', 'nusax_combined_eng_dset', 'nt_label_map.csv'),
    'tweet-senti-multi': ('tsm_test_dset', 'icl_tsm_senti_dset', 'eng', 'tsm_combined_eng_dset', 'tsm_combined_eng_dset', 'tsm_label_map.csv'),
}

def load_dataset(dataset, base_path='./dataset'):
    test_data_name, icl_data_name, xicl_lang, iia_data_name, itc_data_name, ioa_data_name = dataset_mapping[dataset]
    
    ioa_df = pd.read_csv(f'{base_path}/{ioa_data_name}').set_index('lang')
    ioa_df['label_map'] = ioa_df['label_map'].apply(lambda x: json.loads(x))
    
    return (
        datasets.load_from_disk(f'{base_path}/{test_data_name}'),
        datasets.load_from_disk(f'{base_path}/{icl_data_name}'),
        xicl_lang,
        datasets.load_from_disk(f'{base_path}/{iia_data_name}'),
        datasets.load_from_disk(f'{base_path}/{itc_data_name}'),
        ioa_df # (lang, label_map)
    )

if __name__ == '__main__':
    base_path = '../dataset'
    for key in dataset_mapping.keys():
        eval_dset, icl_dset, xicl_lang, iia_dset, itc_dset, ioa_df = load_dataset(key, base_path=base_path)
        print(f'== {key} ==')
        
        print('EVAL DSET')
        print(eval_dset)
        print(list(eval_dset.values())[0][:3])
        print()
        
        print('ICL DSET')
        print(icl_dset)
        print(list(icl_dset.values())[0][:3])
        print()

        print('XICL LANG')
        print(xicl_lang)
        
        print('IIA DSET')
        print(iia_dset)
        print(list(iia_dset.values())[0][:3])
        print()
        
        print('ITC DSET')
        print(itc_dset)
        print(list(itc_dset.values())[0][:3])
        print()

