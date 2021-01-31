import os
import shutil
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from rnnt.model_ctc_polish import BRNNCTC
from rnnt.optim import Optimizer, build_optimizer
from rnnt.dataset_ctc_polish import PolishTrainDataset, int2base
from tensorboardX import SummaryWriter
from rnnt.utils_ctc import AttrDict, init_logger, count_parameters, save_model, computer_cer
from rnnt.ctcdecoder import GreedyDecoder


def train(epoch, config, model, training_data, optimizer, logger, visualizer=None):
    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)

    for step, (regions, inputs, inputs_length, targets, targets_length, rle_bases, rle_bases_length, rles,
               rles_length) in enumerate(
        training_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()
            rle_bases, rle_bases_length, rles, rles_length = rle_bases.cuda(), rle_bases_length.cuda(), rles.cuda(), rles_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        max_rle_bases_length = rle_bases_length.max().item()
        max_rles_length = rles_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]
        rle_bases = rle_bases[:, :max_rle_bases_length]
        rles = rles[:, :max_rles_length]

        if config.optim.step_wise_update:
            optimizer.step_decay_lr()

        optimizer.zero_grad()
        start = time.process_time()
        base_loss, rle_loss = model(inputs, inputs_length, rle_bases, rle_bases_length, rles, rles_length)
        # if epoch >= config.training.first_stage:
        #     loss = rle_loss
        # else:
        #     loss = base_loss
        loss = base_loss + rle_loss

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        loss.backward()

        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm)

        optimizer.step()

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            if visualizer is not None:
                visualizer.add_scalar(
                    'train_loss', loss.item(), optimizer.global_step)
                visualizer.add_scalar(
                    'learn_rate', optimizer.lr, optimizer.global_step)
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end - start))

        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step + 1), end_epoch - start_epoch))


def eval(epoch, config, model, validating_data, logger, visualizer=None):
    rle_decoder = GreedyDecoder(int2base, blank_index=0)
    base_decoder = GreedyDecoder(int2base, blank_index=0)
    model.eval()
    total_loss = 0
    total_dist = 0
    total_rle_dist = 0
    total_word = 0
    total_rle_word = 0
    cer = 0
    rle_cer = 0
    batch_steps = len(validating_data)
    for step, (regions, inputs, inputs_length, targets, targets_length, rle_bases, rle_bases_length, rles,
               rles_length) in enumerate(
        validating_data):

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
        pred_rle_strings = [v.upper() for v in
                            pred_rle_strings]  # 相邻碱基采用flipflop，预测成base时为大小写，需要将小写转换成大写。如：AaAaTtG  ->  AAAATTG
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

        #
        cer = total_dist / total_word * 100
        rle_cer = total_rle_dist / total_rle_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%, Rle CER: %.5f %%' % (epoch, process, cer, rle_cer))

    logger.info('-Validation-Epoch:%4d, AverageCER: %.5f %%, AverageRleCER: %.5f %%' % (epoch, cer, rle_cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)
        visualizer.add_scalar('rle_cer', rle_cer, epoch)


def test(epoch, config, model, test_data, logger, visualizer=None):
    rle_decoder = GreedyDecoder(int2base, blank_index=0)
    base_decoder = GreedyDecoder(int2base, blank_index=0)
    model.eval()
    total_loss = 0
    total_dist = 0
    total_rle_dist = 0
    total_word = 0
    total_rle_word = 0
    cer = 0
    rle_cer = 0
    batch_steps = len(test_data)
    for step, (regions, inputs, inputs_length, targets, targets_length, rle_bases, rle_bases_length, rles,
               rles_length) in enumerate(
        test_data):

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

        #
        cer = total_dist / total_word * 100
        rle_cer = total_rle_dist / total_rle_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Test-Epoch:%d(%.5f%%), CER: %.5f %%, Rle CER: %.5f %%' % (epoch, process, cer, rle_cer))

    logger.info('-Test-Epoch:%4d, AverageCER: %.5f %%, AverageRleCER: %.5f %%' % (epoch, cer, rle_cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)
        visualizer.add_scalar('rle_cer', rle_cer, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell2.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('egs', config.configname, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    num_workers = config.training.num_gpu * 2
    train_dataset = PolishTrainDataset(config.data.train, config.model.max_rle)
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=True, num_workers=num_workers)
    logger.info('Load Train Set!')

    dev_dataset = PolishTrainDataset(config.data.dev, config.model.max_rle)
    validate_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=False, num_workers=num_workers)
    logger.info('Load Dev Set!')

    test_dataset = PolishTrainDataset(config.data.test, config.model.max_rle)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
                                            shuffle=False, num_workers=num_workers)
    logger.info('Load Test Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    model = BRNNCTC(config.model)

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('Loaded model from %s' % config.training.load_model)
    elif config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.load_encoder)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.load_decoder)
            model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.load_decoder)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc))

    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info('Created a %s optimizer.' % config.optim.type)

    if opt.mode == 'continue':
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 0

    # create a visualizer
    if config.training.visualization:
        visual_log = os.path.join(exp_name, 'log')
        visualizer = SummaryWriter(visual_log)
        dev_visualizer = SummaryWriter(os.path.join(visual_log, 'dev'))
        test_visualizer = SummaryWriter(os.path.join(visual_log, 'test'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    for epoch in range(start_epoch, config.training.epochs):
        if epoch == config.training.first_stage:
            # freeze encoder params
            for param in model.encoder.parameters():
                param.requires_grad = False
            # freeze decoder params
            for param in model.decoder.parameters():
                param.requires_grad = False
            for param in model.joint.forward_layer.parameters():
                param.requires_grad = False
            for param in model.joint.base_project_layer.parameters():
                param.requires_grad = False
            optimizer.optimizer = build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), config.optim)

        train(epoch, config, model, training_data, optimizer, logger, visualizer)

        if config.training.eval_or_not:
            _ = eval(epoch, config, model, validate_data, logger, dev_visualizer)
            _ = test(epoch, config, model, test_data, logger, test_visualizer)

        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        save_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)

        if epoch >= config.optim.begin_to_adjust_lr:
            optimizer.decay_lr()
            # early stop
            if optimizer.lr < 1e-6:
                logger.info('The learning rate is too low to train.')
                break
            logger.info('Epoch %d update learning rate: %.6f' %
                        (epoch, optimizer.lr))

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()
