if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

make IFT_GPU=$1 iftCreateGFLIMGraph iftGFLIMConvertSeeds iftGFLIMDecodeLayer iftGFLIMExtract iftGFLIMExtract2 iftGFLIMWriteActivations iftWriteSeedsVectors iftRGBBucketDescriptor iftCreateGFLIMGraphFromLabels iftGraphToMimg iftGFLIMActivDataSet iftGFLIMDataSetActiv
cd GFLIM
bash compile.sh $1
cd ..
cd GFLIM-BoFP
bash compile.sh $1
cd ..
cd GFLIM-BoFP-U
bash compile.sh $1
cd ..
cd M-GFLIM-BoFP
bash compile.sh $1
cd ..

