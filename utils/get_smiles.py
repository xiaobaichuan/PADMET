#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile

import time

millis = int(round(time.time() * 1000))


def iupac_to_smiles(iupac_name, jar_path="./jar/opsin-cli-2.8.0-jar-with-dependencies.jar"):
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp:
        tmp.write(iupac_name)
        tmp.flush()
        tmp_path = tmp.name

    cmd = [
        "/root/anaconda3/envs/padmet/bin/java", "-jar", jar_path,
        "-osmi", tmp_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    os.remove(tmp_path)
    if process.returncode != 0:
        print(f"OPSIN error: {stderr}")
        return ''
    try:
        smiles = stdout.strip().split("\t")[-1]
    except Exception as e:
        print(e)
        smiles = ''
    return smiles


if __name__ == '__main__':
    # get_data('deamino-L-cysteinyl-(2S,3aS,7aS)-octahydroindole-2-carbonyl-L-arginyl-glycyl-DL-alpha-aspartyl-D-tryptophanyl-DL-prolyl-L-Cysteine')
    # print(iupac_to_smiles('N-decanoyl-L-cysteinyl-L-homoarginyl-glycyl-L-alpha-aspartyl-L-tryptophanyl-alpha-methyl-L-prolyl-glycinol'))
    print(iupac_to_smiles('decanoyl-ycinol'))
