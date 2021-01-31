import argparse
import yaml
from tqdm import tqdm
import torch
from rnnt.model_ctc_polish import BRNNCTC
from rnnt.dataset_ctc_polish import PolishTrainDataset, int2base
from rnnt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
from rnnt.ctcdecoder import GreedyDecoder

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.pairwise2 import format_alignment


def calculate_identity(preds, labels):
    match, insert, delete, mismatch = 0, 0, 0, 0
    batch_length = 0
    insert_index_set = set()
    for idx, (label, pred) in enumerate(zip(labels, preds)):
        label = Seq(label)
        pred = Seq(pred)
        alignments = pairwise2.align.globalxx(label, pred)
        if len(alignments) == 0:
            return 0, 0, 0, 0, 0
        aln_label = alignments[0][0]
        aln_pred = alignments[0][1]
        for i in range(len(aln_label)):
            if aln_label[i] == aln_pred[i]:
                match += 1
            elif aln_label[i] != '-' and aln_pred[i] != '-' and aln_label[i] != aln_pred[i]:
                mismatch += 1
            elif aln_label[i] == '-' and aln_pred[i] != '-':
                insert += 1
                insert_index_set.add(idx)
            elif aln_label[i] != '-' and aln_pred[i] == '-':
                delete += 1
            else:
                print("unknown.")
        batch_length += len(label)
    return match, insert, delete, mismatch, batch_length, insert_index_set


def eval_cer(config, model, validating_data):
    rle_decoder = GreedyDecoder(int2base, blank_index=0)
    base_decoder = GreedyDecoder(int2base, blank_index=0)
    model.eval()
    total_loss = 0
    total_dist = 0
    total_rle_dist = 0
    total_word = 0
    total_rle_word = 0
    total_match, total_insert, total_delete, total_mismatch, total_length = 0, 0, 0, 0, 0
    batch_steps = len(validating_data)
    for step, (regions, inputs, inputs_length, targets, targets_length, rle_bases, rle_bases_length, rles,
               rles_length) in enumerate(validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()
            rle_bases, rle_bases_length, rles, rles_length = rle_bases.cuda(), rle_bases_length.cuda(), rles.cuda(), rles_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        max_rle_bases_length = rle_bases_length.max().item()
        max_rles_length = rles_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]  # [N, max_inputs_length, c]
        targets = targets[:, :max_targets_length]  # [N, max_targets_length]
        rle_bases = rle_bases[:, :max_rle_bases_length]
        rles = rles[:, :max_rles_length]

        base_logits, rle_logits = model.recognize(inputs, inputs_length)  # [B,L,o]
        pred_base_strings, offset = base_decoder.decode(base_logits, inputs_length)
        pred_rle_strings, offset = rle_decoder.decode(rle_logits, inputs_length)
        pred_base_strings = [v.upper() for v in pred_base_strings]
        pred_rle_strings = [v.upper() for v in pred_rle_strings]
        # print(pred_base_strings, pred_rle_strings)

        # print('preds')
        # print(preds)

        base_targets = [rle_bases.cpu().numpy()[i][:rle_bases_length[i].item()].tolist()
                        for i in range(rle_bases.size(0))]
        base_transcripts = []
        for i in range(rle_bases.size(0)):
            rle_target_seq = ""
            for v in base_targets[i]:
                rle_target_seq += int2base[v]
            base_transcripts.append(''.join(rle_target_seq))

        rle_targets = [targets.cpu().numpy()[i][:targets_length[i].item()].tolist()
                       for i in range(rles.size(0))]
        rle_transcripts = []
        for i in range(targets.size(0)):
            rle_target_seq = ""
            for v in rle_targets[i]:
                rle_target_seq += int2base[v]
            rle_transcripts.append(''.join(rle_target_seq))

        #         # print('transcripts')
        #         # print(transcripts)
        #
        dist, num_words = computer_cer(pred_base_strings, base_transcripts)
        total_dist += dist
        total_word += num_words
        # print('base:',pred_base_strings, transcripts)

        rle_dist, rle_num_words = computer_cer(pred_rle_strings, rle_transcripts)
        total_rle_dist += rle_dist
        total_rle_word += rle_num_words
        # print('rle:',pred_rle_strings, rle_transcripts)
        # print()

        match, insert, delete, mismatch, batch_length, insert_set = calculate_identity(pred_rle_strings,
                                                                                       rle_transcripts)
        for idx in insert_set:
            print(regions[idx])
            print("pred :", pred_rle_strings[idx])
            print("label:", rle_transcripts[idx])
            print(inputs.cpu()[idx][:inputs_length[idx]][:, :7])
            print('-' * 10)

        total_match += match
        total_insert += insert
        total_delete += delete
        total_mismatch += mismatch
        total_length += batch_length
        #
        cer = total_dist / total_word * 100
        rle_cer = total_rle_dist / total_rle_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            print('-Validation:(%.5f%%), CER: %.5f %%, Rle CER: %.5f %%' % (process, cer, rle_cer))

    val_loss = total_loss / (step + 1)
    print(
        '-Validation:, AverageLoss:%.5f, AverageCER: %.5f %%, AverageRleCER: %.5f %%, Mismatch: %.5f %%, Insertion: %.5f %%, Deletion: %.5f %%' %
        (val_loss, cer, rle_cer, total_mismatch / total_length * 100, total_insert / total_length * 100,
         total_delete / total_length * 100))


def eval(config, model, validating_data):
    rle_decoder = GreedyDecoder(int2base, blank_index=0)
    base_decoder = GreedyDecoder(int2base, blank_index=0)

    model.eval()
    for (
            regions, inputs, inputs_length, targets, targets_length, rle_bases, rle_bases_length, rles,
            rles_length) in tqdm(
        validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()
            rle_bases, rle_bases_length, rles, rles_length = rle_bases.cuda(), rle_bases_length.cuda(), rles.cuda(), rles_length.cuda()

        batch_size = inputs.shape[0]
        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        max_rle_bases_length = rle_bases_length.max().item()
        max_rles_length = rles_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]  # [N, max_inputs_length, c]
        targets = targets[:, :max_targets_length]  # [N, max_targets_length]
        rle_bases = rle_bases[:, :max_rle_bases_length]
        rles = rles[:, :max_rles_length]

        base_logits, rle_logits = model.recognize(inputs, inputs_length)  # [B,L,o]
        pred_base_strings, offset = base_decoder.decode(base_logits, inputs_length)
        pred_rle_strings, offset = rle_decoder.decode(rle_logits, inputs_length)
        pred_base_strings = [v.upper() for v in pred_base_strings]
        pred_rle_strings = [v.upper() for v in pred_rle_strings]

        base_targets = [rle_bases.cpu().numpy()[i][:rle_bases_length[i].item()].tolist()
                        for i in range(rle_bases.size(0))]
        base_transcripts = []
        for i in range(rle_bases.size(0)):
            rle_target_seq = ""
            for v in base_targets[i]:
                rle_target_seq += int2base[v]
            base_transcripts.append(''.join(rle_target_seq))

        rle_targets = [targets.cpu().numpy()[i][:targets_length[i].item()].tolist()
                       for i in range(rles.size(0))]
        rle_transcripts = []
        for i in range(targets.size(0)):
            rle_target_seq = ""
            for v in rle_targets[i]:
                rle_target_seq += int2base[v]
            rle_transcripts.append(''.join(rle_target_seq))

        for i in range(batch_size):
            print(regions[i], " ,base preds:", pred_base_strings[i])
            print(regions[i], " ,base label:", base_transcripts[i])
            print(regions[i], " ,rle  preds:", pred_rle_strings[i])
            print(regions[i], " ,rle  label:", rle_transcripts[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-data', type=str, required=True)
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    dev_dataset = PolishTrainDataset(opt.data, config.model.max_rle)
    validate_data = torch.utils.data.DataLoader(dev_dataset, batch_size=512, shuffle=False, num_workers=20)

    model = BRNNCTC(config.model)

    checkpoint = torch.load(opt.model)
    model.load_state_dict(checkpoint['forward'])

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    eval(config, model, validate_data)
    # eval_cer(config, model, validate_data)


if __name__ == '__main__':
    main()
