import re


def naive_parse_paracrawl_tmx(file_in, file_out):
    print(f"Parsing {file_in}")

    with open(file_in) as f:
        tmx_file = f.read()

    print("You've got enough memory! Congrats! Now lets see how patient you are.")

    srcl, tgtl = 'en', 'et'

    TU = re.compile(r'<tu (.*?)</tu>', re.DOTALL)
    TUV_SRC = re.compile(fr'<tuv xml:lang="{srcl}">(.*?)</tuv>', re.DOTALL)
    TUV_TGT = re.compile(fr'<tuv xml:lang="{tgtl}">(.*?)</tuv>', re.DOTALL)
    SEG = re.compile(fr'<seg>(.*?)</seg>', re.DOTALL)
    DOC = re.compile(fr'<prop type="source-document">(.*?)</prop>', re.DOTALL)
    
    tus = re.findall(TU, tmx_file)
    
    with open(file_out, "w") as f:
        for tu in tus:
            src_tuv = re.findall(TUV_SRC, tu)
            tgt_tuv = re.findall(TUV_TGT, tu)
            assert len(src_tuv) == 1
            assert len(tgt_tuv) == 1

            src_tuv, tgt_tuv = src_tuv[0].strip(), tgt_tuv[0].strip()

            doc_id = re.findall(DOC, src_tuv)[0]
            src_sent, tgt_sent = re.findall(SEG, src_tuv), re.findall(SEG, tgt_tuv)

            assert len(src_sent) == 1
            assert len(tgt_sent) == 1

            src_sent, tgt_sent = src_sent[0], tgt_sent[0]

            f.write(f"{doc_id}\t{src_sent}\t{tgt_sent}\n")

    print(f"Saved to {file_out}")


if __name__ == '__main__':
    ddir = "experiments/en-et_paracrawl"
    naive_parse_paracrawl_tmx(f"{ddir}/en-et.tmx", f"{ddir}/doc-indices-paracrawl.txt")