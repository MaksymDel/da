import os
from os.path import basename, dirname
from collections import defaultdict
import sys

def split_by_clust(exp, langpair):
    l1, l2 = langpair.split("-")

    if exp == "bert":
        model_name = "xlm-roberta-base"
    elif exp == "nmt":
        model_name = "concat60"
    else:
        raise ValueError(f"exp")
    
    for level in ["sent", "doc"]:

        clust_folder = f"experiments/{l1}_{l2}_{model_name}/{exp}-clusters-{level}"
        data_folder = "experiments/fairseq-data-en-et-base"

        for split in ["test", "valid", "train"]:
            if split == "valid":
                split2 = "dev"
            else:
                split2 = split

            fnc = f"{clust_folder}/base.{split2}.clust.{exp}.{level}"

            with open(fnc) as f:
                clust = f.read().splitlines()
            with open(f"{data_folder}/{split}.{l1}") as f:
                d1 = f.read().splitlines()
            with open(f"{data_folder}/{split}.{l2}") as f:
                d2 = f.read().splitlines()
            assert len(clust) == len(d1)

            d1_split = defaultdict(list)
            d2_split = defaultdict(list)

            for i, c in enumerate(clust):
                d1_split[c].append(d1[i])
                d2_split[c].append(d2[i])


            dirout_base = f"{clust_folder}-data"
            os.makedirs(dirout_base, exist_ok=True)


            for k, v in d1_split.items():
                dirout = f"{dirout_base}/{k}"
                os.makedirs(dirout, exist_ok=True)

                with open(f"{dirout}/{split}.{l1}", "w") as f:
                    f.write("\n".join(map(str, v)))


            for k, v in d2_split.items():
                dirout = f"{dirout_base}/{k}"
                os.makedirs(dirout, exist_ok=True)

                with open(f"{dirout}/{split}.{l2}", "w") as f:
                        f.write("\n".join(map(str, v)))



if __name__ == '__main__':
    exp = sys.argv[1]
    langpair = sys.argv[2]
    
    split_by_clust(exp, langpair)
