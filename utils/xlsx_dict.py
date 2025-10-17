from collections import OrderedDict

import pandas as pd

input_file = "./utils/modification.txt"

df = pd.read_csv(input_file, index_col=0, sep='\t')
modification_config = {
    'residue': {residue: [] for residue in ['C', 'S', 'T', 'R', 'K', 'Y', 'H', 'D', 'E', 'N', 'M', 'F', 'W', 'A', 'I', 'L', 'Q', 'V', 'P', 'G']},
    'position': {
        'N-terminal': [],
        'C-terminal': []
    }
}


def get_modification_config():
    # modifications = df.iloc[:, 0].tolist()
    modifications = list(set(df.index.tolist()))

    # print(len(modifications))
    # print(len(set(modifications)))
    modification_dict = OrderedDict((modification, index) for index, modification in enumerate(modifications))
    for modification, row in df.iterrows():
        sites = row[row == 1].index.tolist()
        for site in sites:
            if site in modification_config['residue']:
                modification_config['residue'][site].append(modification)  # 将修饰名称添加到对应的残基中
            elif site in modification_config['position']:
                modification_config['position'][site].append(modification)  # 将修饰名称添加到对应的位点中
    # print(modification_config)
    # print(modification_dict)
    # print(f'modification_dict of length {len(modification_dict)}:\n{modification_dict}')
    return modification_dict, modification_config


if __name__ == '__main__':
    get_modification_config()
