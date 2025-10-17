import re
amino_acid= {
    "A": "Alanine",
    "R": "Arginine",
    "N": "Asparagine",
    "D": "Aspartic Acid",
    "C": "Cysteine",
    "E": "Glutamic Acid",
    "Q": "Glutamine",
    "G": "Glycine",
    "H": "Histidine",
    "I": "Isoleucine",
    "L": "Leucine",
    "K": "Lysine",
    "M": "Methionine",
    "F": "Phenylalanine",
    "P": "Proline",
    "S": "Serine",
    "T": "Threonine",
    "W": "Tryptophan",
    "Y": "Tyrosine",
    "V": "Valine"
}

res_monomer={
    "alanyl": "Alanine",
    "arginyl": "Arginine",
    "asparaginyl": "Asparagine",
    "aspartyl": "Aspartic Acid",
    "cysteinyl": "Cysteine",
    "glutamyl": "Glutamic Acid",
    "glutaminyl": "Glutamine",
    "glycyl": "Glycine",
    "histidyl": "Histidine",
    "isoleucyl": "Isoleucine",
    "leucyl": "Leucine",
    "lysyl": "Lysine",
    "methionyl": "Methionine",
    "phenylalanyl": "Phenylalanine",
    "prolyl": "Proline",
    "seryl": "Serine",
    "threonyl": "Threonine",
    "tryptophanyl": "Tryptophan",
    "tyrosyl": "Tyrosine",
    "valyl": "Valine"
}
amino_acid_residues = {
    "A": "L-alanyl", "R": "L-arginyl", "N": "L-asparaginyl", "D": "L-alpha-aspartyl",
    "C": "L-cysteinyl", "E": "L-alpha-glutamyl", "Q": "L-glutaminyl", "G": "glycyl",
    "H": "L-histidyl", "I": "L-isoleucyl", "L": "L-leucyl", "K": "L-lysyl",
    "M": "L-methionyl", "F": "L-phenylalanyl", "P": "L-prolyl", "S": "L-seryl",
    "T": "L-threonyl", "W": "L-tryptophanyl", "Y": "L-tyrosyl", "V": "L-valyl",
}

amino_acid_residues_rev = {v.split('-')[-1]: k for k, v in amino_acid_residues.items()}

substitutions = {
    "A": {'>>DL-alanyl':'DL-alanyl','>>D-alanyl':'D-alanyl','>>beta-alanyl':'beta-alanyl','>L-alaninamide': 'L-alaninamide',},
    "R": {'>>DL-arginyl':'DL-arginyl', '>>D-arginyl':'D-arginyl' ,'>homoarginyl': 'L-homoarginyl','>L-argininamide':'L-argininamide','>D-argininamide':'D-argininamide',},
    "N": {'>>DL-asparaginyl':'DL-asparaginyl','>>D-asparaginyl':'D-asparaginyl'},
    "D": {'>>DL-alpha-aspartyl':'DL-alpha-aspartyl','>>D-alpha-aspartyl':'D-alpha-aspartyl'},
    "C": {'>>DL-cysteinyl':'DL-cysteinyl','>>D-cysteinyl':'D-cysteinyl','>L-cysteinamide': 'L-cysteinamide', },
    "E": {'>>DL-alpha-glutamyl':'DL-alpha-glutamyl','>>D-alpha-glutamyl':'D-alpha-glutamyl','>pyroglutamic acid':'L-pyroglutamyl',},
    "Q": {'>>DL-glutaminyl':'DL-glutaminyl','>>L-gamma-glutamyl':'L-gamma-glutamyl','>pyroglutamic acid':'L-pyroglutamyl',},
    "G": {'>glycinamide': 'glycinamide','>glycinol':'glycinol',},
    "H": {'>>DL-histidyl':'DL-histidyl','>>D-histidyl':'D-histidyl'},
    "I": {'>>DL-isoleucyl':'DL-isoleucyl','>>D-isoleucyl':'D-isoleucyl'},
    "L": {'>>DL-leucyl':'DL-leucyl','>>D-leucyl':'D-leucyl','>L-lysinamide':'L-lysinamide','>L-norleucyl':'3L-norleucyl',},
    "K": {'>>DL-lysyl':'DL-lysyl','>>D-lysyl':'D-lysyl'},
    "M": {'>>DL-methionyl':'DL-methionyl','>>D-methionyl':'D-methionyl'},
    "F": {'>>DL-phenylalanyl':'DL-phenylalanyl','>>D-phenylalanyl':'D-phenylalanyl','>phenylalaninamide':'phenylalaninamide','>pyroglutamic acid':'L-pyroglutamyl',},
    "P": {'>>DL-prolyl':'DL-prolyl','>>D-prolyl':'D-prolyl','>L-proline ethylamide':'L-proline ethylamide','>L-prolinamide':'L-prolinamide',},
    "S": {'>>DL-seryl':'DL-seryl','>>D-seryl':'D-seryl','>L-serinamide':'L-serinamide','>L-prolinehydrazide':'L-prolinehydrazide'},
    "T": {'>>DL-threonyl':'DL-threonyl','>>D-threonyl':'D-threonyl','>L-threoninamide':'L-threoninamide','>L-threoninol':'L-threoninol',},
    "W": {'>>DL-tryptophanyl':'DL-tryptophanyl','>>D-tryptophanyl':'D-tryptophanyl'},
    "Y": {'>>DL-tyrosyl':'DL-tyrosyl','>>D-tyrosyl':'D-tyrosyl','>L-tyrosinamide':'L-tyrosinamide',},
    "V": {'>>DL-valyl':'DL-valyl','>>D-valyl':'D-valyl','>L-valinamide':'L-valinamide',}
}


modifcation_map = {
    'deamino': 'deamino', 'acetyl':'acetyl', '3-(2-naphthyl)': '3-(2-naphthyl)',
    '3-(3-pyridyl) modification':'3-(3-pyridyl)','para-chloro modification':'4-chloro',
    'N-methylation':'N-methyl','N6-isopropyl modification':'N6-isopropyl','formyl':'formyl',
    "N'-Carbamoylation":"N'-carbamoyl",
    'O-tert-butyl':'O-tert-butyl','1-benzyl':'1-benzyl', 'O4-ethyl':'O4-ethyl',
    '4-((S)-dihydroorotamido)':'4-((S)-dihydroorotamido)','4-ureido':'4-ureido',
    'O4-sulfo':'O4-sulfo','alpha-methyl':'alpha-methyl','4-fluorobenzoyl':'(4-fluorobenzoyl)',
    'Boc':'N-tert-butoxycarbonyl','(2S,3aS,7aS)-octahydroindole-2-carbonyl':'(2S,3aS,7aS)-octahydroindole-2-carbonyl',
    '(3R)-1,2,3,4-tetrahydroisoquinoline-3-carbonyl':'(3R)-1,2,3,4-tetrahydroisoquinoline-3-carbonyl',
    'decanoyl':'decanoyl','thienyl':'2-thienyl','No-mod': '',}

amino_acid_residues.update(modifcation_map)

def sequence_to_residues(sequence):
    """将氨基酸序列转换为标准残基形式"""
    return [amino_acid_residues[aa] for aa in sequence]

def add_modifications(sequence, modifications):
    """应用修饰规则生成最终IUPAC名称"""
    residues = sequence_to_residues(sequence)
    # print(residues)
    # print(modifications.keys())
    # assert len(sequence) == len(modifications), f"length error: {len(sequence)}, {len(modifications)}."
    for position in sorted(modifications.keys()):
        modification = modifications[position]
        # print(position, modification)
        # idx = position - 1
        # if modification == 'No-mod':
            # continue
        # 手性替换（>>D）
        if modification.startswith(">>"):
            chiral = modification[2:]
            key = f">>{chiral}"
            if key in substitutions[sequence[position]]:
                residues[position] = substitutions[sequence[position]][key]
            else:
                # 默认替换L-为D-
                residues[position] = residues[position].replace("L-", f"{chiral}-")
            # print(1, residues[position])

        # 氨基酸替换（>homoarginyl等）
        elif modification.startswith(">"):
            key = modification
            if key in substitutions[sequence[position]]:
                residues[position] = substitutions[sequence[position]][key]
            else:
                # 直接替换残基名
                base = residues[position].split("-", 1)[0]  # 如L
                residues[position] = f"{modification[1:]}" if '-' in modification else f"{base}-{modification[1:]}"
            # print(2, residues[position])
            
        # 其他修饰
        elif amino_acid_residues.get(modification) and position:
            residues[position] = f"{amino_acid_residues[modification]}-{residues[position]}"
            # print(3, residues[position])
        
        
        # print(f"{position}/{len(residues)}")
        # N端修饰
        if position == 0 and amino_acid_residues.get(modification):
            if modification == 'deamino':
                residues[0] = f"{amino_acid_residues[modification]}-{residues[0]}"
            else:
                residues[0] = f"N-{amino_acid_residues[modification]}-{residues[0]}"
            
        # C端修饰
        if position == len(residues)-1 and substitutions[sequence[-1]].get(modification):
            residues[-1] = f"{substitutions[sequence[-1]][modification]}"
            suffix = residues[-1].split('-')
            # print(4, suffix[:-1])
            if suffix[-1] in amino_acid_residues_rev:
                
                # print(amino_acid[amino_acid_residues_rev[suffix[-1]]])
                residues[-1] = '-'.join(suffix[:-1]+[amino_acid[amino_acid_residues_rev[suffix[-1]]]])
        
        # elif position == len(residues)-1:
            # print(position)
            
                # print(residues[-1])
            # else:  
                # residues[-1] = f"L-{amino_acid[sequence[-1]]}"
        # print(residues)
        result = "-".join(residues)
        pattern = re.compile(r'(L|D|DL)-(?=[gG]ly)')
        result = re.sub(pattern, '', result)
        
            # **新增逻辑：检查最后一个单词是否为氨基酸残基并替换为单体形式**
        last_dash_index = max(result.rfind('-'), result.rfind(' '))  # 同时查找最后一个“-”或空格
        if last_dash_index != -1:
            last_word = result[last_dash_index + 1:]
            if last_word in res_monomer:  # 判断是否为氨基酸残基
                monomer_name = res_monomer[last_word]  # 获取氨基酸单体形式
                result = result[:last_dash_index + 1] + monomer_name
    return result

# 测试用例
if __name__ == "__main__":
    print(amino_acid_residues_rev)
    test_sequence = "CRGDWPC"
    # test_modifications = {idx: "" for idx in range(len(test_sequence) + 2)}
    test_modifications = {0: '>cysteinamide', 1: '>homoarginyl', 2: '>glycinol', 3: 'No-mod', 4: '>>D-tryptophanyl', 5: '>L-proline ethylamide', 6: 'No-mod'}
    # test_modifications[-1] = '>amide' 
   
  
    result = add_modifications(test_sequence, test_modifications)
    print(f"Modified IUPAC Name: {result}")

