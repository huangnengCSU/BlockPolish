## BlockPolish ##
BlockPolish: accurate polishing of long-read assembly via block divide-and-conquer

## Overview ##
BlockPolish couples four Bidirectional LSTM layers with a compressed projection layer and a flip-flop projection layer to predict the consensus sequence according to the reads-to-assembly alignment. The Bi-LSTM layers take both left and right alignment features when making decisions. The compressed projection layer converts the alignment features to the DNA sequence without continuously repeated nucleotides. The flip-flop projection layer converts the alignment features into the DNA sequence in which the continuous repeated nucleotides are ﬂip-ﬂopped. The ﬂip-ﬂop operation alternately represents the continuously repeated bases using uppercase and lowercase characters (e.g., “AAAAA” is represented as “AaAaA” and “AAATTCT” is represented as “AaATtCT”).

Before neural network-based predicting, the draft assembly is divided into trivial blocks and complex blocks according to reads-to-assembly alignment. The input data of neural network is a sequence of alignment features, which contains percentages of different bases, insertions, and deletions at each position in the block. Dividing contigs and generating feature matrix is done in the `BPFGM` (https://github.com/huangnengCSU/BPFGM).

## Installation ##
Using this method requires the user to install several tools:
- [minimap2](https://github.com/lh3/minimap2)
- [samtools](https://github.com/samtools/samtools)
- [Racon](https://github.com/isovic/racon)
- [BPFGM](https://github.com/huangnengCSU/BPFGM.git)

dependencies:
```
pip install pyyaml editdistance python-Levenshtein biopython tensorboardX
```

pytorch-gpu:
```
```

pytorch-cpu:
```
```

## Usage ##
### Step 1: Run one round of Racon
In the workflow, we recommend to run one round of Racon to polish the draft assembly initially. 
```
minimap2 -x map-ont assembly.fa reads.fq -t 40 > reads2asm.paf
racon reads.fq reads2asm.paf assembly.fa -t 40 > racon_cons0.fasta
```
### Step 2: Align raw reads to Racon polished assembly
```
minimap2 -ax map-ont racon_cons0.fasta reads.fq -t 40 > reads2racon.sam
samtools view -bS -@ 40 reads2racon.sam -o reads2racon.bam
samtools sort -@ 40 reads2racon.bam -o reads2racon.sorted.bam
samtools index -@ 40 reads2racon.sorted.bam
```
### Step 3: Dividing draft assembly and generating feature matrices
In this step, the draft assembly is divided into trivial blocks and complex blocks with different qualities according to reads-to-assembly alignment. Then the feature matrix of each block is extracted including percentages of different bases, insertions, and deletions at each position.
```
block -b reads2racon.sorted.bam -s trivial_features.txt -c complex_features.txt
```
### Step 4: Polishing trivial blocks and complex blocks
The config files and model files are released on Google Drive https://drive.google.com/drive/folders/1JVIANm7ZdGI27ZMJ_TXn_TwKPmXmQPPh.
```
python brnnctc_generate.py -config config/test_trivial_config.yaml -model trivial_model.chkpt -data trivial_features.txt -output trivial_polished.txt

python brnnctc_generate.py -config config/test_complex_config.yaml -model complex_model.chkpt -data complex_features.txt -output complex_polished.txt

python needle.py --t trivial_polished.txt --c complex_polished.txt -output polished_assembly.fa
```

## License ##
Copyright (C) 2020 by Huangneng (huangn94@foxmail.com)