import re
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def chunk_parse_paracrawl_tmx(file_in, file_out,
                              src_lang, tgt_lang, logging_freq):
    srcl, tgtl = src_lang, tgt_lang

    TU = re.compile(r'<tu (.*?)</tu>', re.DOTALL)
    TUV_SRC = re.compile(fr'<tuv xml:lang="{srcl}">(.*?)</tuv>', re.DOTALL)
    TUV_TGT = re.compile(fr'<tuv xml:lang="{tgtl}">(.*?)</tuv>', re.DOTALL)
    SEG = re.compile(fr'<seg>(.*?)</seg>', re.DOTALL)
    DOC = re.compile(fr'<prop type="source-document">(.*?)</prop>', re.DOTALL)

    logging.info(f"Parsing {file_in}")

    with open(file_in, 'r', encoding='utf8') as in_fh, \
            open(file_out, 'w', encoding='utf8') as out_fh:
        chunks_written = 0
        chunk = ""
        line = in_fh.readline()
        while line:
            if re.match(r'<tu tuid=.+>', line.strip()):
                chunk = line
            elif re.match(r'</tu>', line.strip()):
                chunk += line

                assert(re.fullmatch(TU, chunk.strip()))

                src_tuv = re.findall(TUV_SRC, chunk)
                tgt_tuv = re.findall(TUV_TGT, chunk)
                assert len(src_tuv) == 1
                assert len(tgt_tuv) == 1

                src_tuv, tgt_tuv = src_tuv[0].strip(), tgt_tuv[0].strip()

                doc_id = re.findall(DOC, src_tuv)[0]
                src_sent, tgt_sent = re.findall(SEG, src_tuv), re.findall(SEG,
                                                                          tgt_tuv)

                assert len(src_sent) == 1
                assert len(tgt_sent) == 1

                src_sent, tgt_sent = src_sent[0], tgt_sent[0]

                out_fh.write(f"{doc_id}\t{src_sent}\t{tgt_sent}\n")
                chunks_written += 1
                if chunks_written % logging_freq == 0:
                    logging.info(f"Processed {chunks_written} sentence pairs")

                chunk = ""
            else:
                chunk += line

            line = in_fh.readline()

    logging.info(f"Saved to {file_out}, {chunks_written} sentence pairs")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ddir", type=str, required=True,
                        help="Directory containing the input .tmx file",
                        default="experiments/en-et_paracrawl")
    parser.add_argument("--input", type=str, required=True,
                        help="Input .tmx file",
                        default="en-et.tmx")
    parser.add_argument("--output", type=str,
                        help="""Output file (will be written into the same 
                                location as the input file)""",
                        default="doc-indices-paracrawl.txt")
    parser.add_argument("--logfreq", type=int,
                        help="Log progress every N sentence pairs",
                        default=1000000)
    parser.add_argument("--srcl", type=str,
                        help="Source language", default='en')
    parser.add_argument("--tgtl", type=str,
                        help="Target language", default='et')

    args = parser.parse_args()

    chunk_parse_paracrawl_tmx(f"{args.ddir}/{args.input}",
                              f"{args.ddir}/{args.output}",
                              args.srcl, args.tgtl, args.logfreq)
