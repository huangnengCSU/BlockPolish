#!/usr/bin/env python
import argparse


class Record():
    def __init__(self, refName, start, end, id, seq):
        self.refName = refName
        self.start = int(start)
        self.end = int(end)
        self.id = int(id)
        self.seq = seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--t", type=str, required=True, help="trivial region prediction file")
    parser.add_argument("-c","--c", type=str, required=True, help="complex region prediction file")
    parser.add_argument("-output", type=str, required=True, help="output polished file")
    opt = parser.parse_args()

    pieces = {}  # 二级字典，第一级的key=contig_name

    fout = open(opt.output,'w')

    with open(opt.t) as fin:
        for line in fin:
            # print(line)
            try:
                refName, start, end, id, seq = line.rstrip().split('\t')
            except:
                refName, start, end, id = line.rstrip().split('\t')
                seq = ''
            rd = Record(refName, start, end, id, seq)
            pieces[refName] = pieces.get(refName, [])
            pieces[refName].append(rd)

    with open(opt.c) as fin:
        for line in fin:
            # print(line)
            try:
                refName, start, end, id, seq = line.rstrip().split('\t')
            except:
                refName, start, end, id = line.rstrip().split('\t')
                seq = ''
            rd = Record(refName, start, end, id, seq)
            pieces[refName] = pieces.get(refName, [])
            pieces[refName].append(rd)

    for refName in pieces.keys():
        records = pieces[refName]
        sorted_records = sorted(records, key=lambda v: (v.start, v.id), reverse=False)
        contig_seq = ""
        for v in sorted_records:
            contig_seq+=v.seq
        fout.write('>'+refName+'\n')
        fout.write(contig_seq+'\n')

    fout.close()


