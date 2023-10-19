import os, sys
from os.path import exists
import glob
import datasets

# NusaWrites Data Loading
dataset_mapping = {
    'americasnli': ('americasnli_test_dset', 'americasnli_combined_dev_dset', 'americasnli_label_map'),
    'nusatranslation': ('nt_senti_test_dset', 'nusax_combined_ind_dset', 'nt_label_map'),
    'masakhanews': ('masakhanews_test_dset', 'mafand_rand_label_dset', 'masakhanews_label_map')
}

def load_dataset(dataset, base_path='./dataset'):
    test_data_name, icl_data_name, label_map_name = dataset_mapping[dataset]
    return (
        datasets.load_from_disk(f'{base_path}/{test_data_name}'),
        datasets.load_from_disk(f'{base_path}/{icl_data_name}')
        # TODO: Load label mapping
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

