import os, sys
from os.path import exists
import glob
import pandas as pd
import datasets

# NusaWrites Data Loading
dataset_mapping = { 
    # key: (test_dset, icl_dset, x-icl_lang, iia_dset, itc_dset, ioa_dset)
    'americasnli': ('americasnli_test_dset', 'icl_americasnli_dset', 'spa', 'americasnli_combined_dev_dset', 'americasnli_combined_dev_dset', 'americasnli_label_map.csv')
    'masakhanews': ('masakhanews_test_dset', 'icl_masakhanews_dset', 'eng', 'mafand_mt_dset', 'mafand_rand_label_dset', 'masakhanews_label_map.csv')
    'nusatranslation': ('nt_senti_test_dset', 'icl_nusax_senti_dset', 'eng', 'nusax_mt_eng_dset', 'nusax_combined_eng_dset', 'nt_label_map.csv')
}

def load_dataset(dataset, base_path='./dataset'):
    test_data_name, icl_data_name, xicl_lang, iia_data_name, itc_data_name, ioa_data_name = dataset_mapping[dataset]
    return (
        datasets.load_from_disk(f'{base_path}/{test_data_name}'),
        datasets.load_from_disk(f'{base_path}/{icl_data_name}'),
        xicl_lang,
        datasets.load_from_disk(f'{base_path}/{iia_data_name}'),
        datasets.load_from_disk(f'{base_path}/{itc_data_name}'),
        pd.read_csv(f'{base_path}/{ioa_data_name}').set_index('lang') # (lang, label_map)
    )

if __name__ == '__main__':
    base_path = '../dataset'
    for key in dataset_mapping.keys():
        test_dset, icl_dset = load_dataset(key, base_path=base_path)
        print(f'== {key} ==')
        
        print('TEST DSET')
        print(test_dset)
        print(list(test_dset.values())[0][:3])
        print()
        
        print('ICL DSET')
        print(icl_dset)
        print(list(icl_dset.values())[0][:3])
        print()

