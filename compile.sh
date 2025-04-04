if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

make IFT_GPU=$1 iftCreateGFLIMGraph iftGFLIMConvertSeeds iftGFLIMDecodeLayer iftGFLIMExtract iftGFLIMWriteActivations
cd GFLIM
bash compile.sh $1
cd ..
cd GFLIM-BoFP
bash compile.sh $1
cd ..
