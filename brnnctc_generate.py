#!/usr/bin/env python
import argparse
import yaml
from tqdm import tqdm
import torch
import math
from rnnt.model_ctc_polish import BRNNCTC
from rnnt.dataset_ctc_polish import PolishGenerateDataset, int2base, PolishPredictDataset
from rnnt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
from rnnt.ctcdecoder import GreedyDecoder


def generate(config, model, validating_data, outfile, device):
    rle_decoder = GreedyDecoder(int2base, blank_index=0)
    base_decoder = GreedyDecoder(int2base, blank_index=0)

    model.eval()
    region_results = []
    for (regions, inputs, inputs_length) in tqdm(validating_data):
        inputs = inputs[0]
        inputs_length = inputs_length[0]

        ## split feat if too big
        if inputs.shape[0] > 1000:
            split_num = math.ceil(inputs.shape[0] / 1000)
            for j in range(split_num):
                if j == split_num - 1:
                    inputs_split = inputs[j * 1000:]
                    inputs_length_split = inputs_length[j * 1000:]
                    regions_split = regions[j * 1000:]
                else:
                    inputs_split = inputs[j * 1000: (j + 1) * 1000]
                    inputs_length_split = inputs_length[j * 1000: (j + 1) * 1000]
                    regions_split = regions[j * 1000: (j + 1) * 1000]

                inputs_split = inputs_split.to(device)

                batch_size = inputs_split.shape[0]
                max_inputs_length_split = inputs_length_split.max().item()
                inputs_split = inputs_split[:, :max_inputs_length_split, :]

                base_logits, rle_logits = model.recognize(inputs_split, inputs_length_split)  # [B,L,o]
                pred_rle_strings, offset = rle_decoder.decode(rle_logits, inputs_length_split)
                pred_rle_strings = [v.upper() for v in pred_rle_strings]
                
                for i in range(batch_size):
                    region_results.append([regions_split[i][0], pred_rle_strings[i]])
        else:
            inputs = inputs.to(device)
            # inputs_length = inputs_length.to(device)

            batch_size = inputs.shape[0]
            max_inputs_length = inputs_length.max().item()
            inputs = inputs[:, :max_inputs_length, :]  # [N, max_inputs_length, c]

            base_logits, rle_logits = model.recognize(inputs, inputs_length)  # [B,L,o]
            # pred_base_strings, offset = base_decoder.decode(base_logits, inputs_length)
            pred_rle_strings, offset = rle_decoder.decode(rle_logits, inputs_length)
            pred_rle_strings = [v.upper() for v in pred_rle_strings]

            for i in range(batch_size):
                # region_results.append([regions[i], pred_rle_strings[i]])
                region_results.append([regions[i][0], pred_rle_strings[i]])

        if len(region_results) > 10000:
            for record in region_results:
                outfile.write(record[0] + "\t" + record[1] + "\n")
            region_results = []
    if len(region_results) > 0:
        for record in region_results:
            outfile.write(record[0] + "\t" + record[1] + "\n")
        region_results = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-output', type=str, required=True)
    parser.add_argument('--no_cuda', action="store_true", help='If running on cpu device, set the argument.')
    opt = parser.parse_args()
    device = torch.device('cuda' if not opt.no_cuda else 'cpu')

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    fout = open(opt.output, 'w')

    # dev_dataset = PolishGenerateDataset(opt.data)
    # validate_data = torch.utils.data.DataLoader(dev_dataset, batch_size=512, shuffle=False, num_workers=20)
    # dev_dataset = PolishGenerateDataset(opt.data, 1024, 40)
    # validate_data = torch.utils.data.DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    dev_dataset = PolishPredictDataset(opt.data)
    validate_data = torch.utils.data.DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = BRNNCTC(config.model).to(device)
    checkpoint = torch.load(opt.model, map_location=device)
    model.load_state_dict(checkpoint['forward'])

    # if config.training.num_gpu > 0:
    #     model = model.cuda()
    #     if config.training.num_gpu > 1:
    #         device_ids = list(range(config.training.num_gpu))
    #         model = torch.nn.DataParallel(model, device_ids=device_ids)
    generate(config, model, validate_data, fout, device)
    fout.close()


if __name__ == '__main__':
    main()
