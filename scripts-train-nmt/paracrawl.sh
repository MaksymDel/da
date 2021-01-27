#cd experiments/en-et_paracrawl
#wget https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-et.tmx.gz
#gunzip en-et.tmx.gz
#cd ../..
python scripts-train-nmt/parse_paracrawl_tmx.py
