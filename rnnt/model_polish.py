import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.encoder import build_encoder
from rnnt.decoder import build_decoder
import warp_rnnt._C as core  # 使用`https://github.com/1ytic/warp-rnnt`的rnntloss
from rnnt.rnntloss import rnnt_loss
import copy


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size, max_rle):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.base_project_layer = nn.Linear(inner_dim, vocab_size, bias=True)
        self.rle_project_layer = nn.Linear(inner_dim, max_rle, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)  # [N, 1, L+1, o1]
            enc_state = enc_state.unsqueeze(2)  # [N, L, 1, o2]

            t = enc_state.size(1)  # L
            u = dec_state.size(2)  # L+1

            enc_state = enc_state.repeat([1, 1, u, 1])  # [N,L,L+1,o1]
            dec_state = dec_state.repeat([1, t, 1, 1])  # [N,L,L+1,o2]
        else:
            assert enc_state.dim() == dec_state.dim()
        # print(enc_state.shape, dec_state.shape)
        concat_state = torch.cat((enc_state, dec_state), dim=-1)  # [N,L,L+1,o1+o2]
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        base_outputs = self.base_project_layer(outputs)  # [N,L,L+1,vocab_size]
        rle_outputs = self.rle_project_layer(outputs)

        return base_outputs, rle_outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(config)
        # define decoder
        self.decoder = build_decoder(config)
        # define JointNet
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size,
            max_rle=config.max_rle
        )

        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.base_project_layer.weight.size(), '%d != %d' % (
                self.decoder.embedding.weight.size(1), self.joint.base_project_layer.weight.size(1))
            self.joint.base_project_layer.weight = self.decoder.embedding.weight

    def forward(self, inputs, inputs_length, targets, targets_length, rles):
        # inputs: N x L x c
        # targets: N x L
        enc_state, _ = self.encoder(inputs, inputs_length)  # [N, L, o]
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=0)  # [N, L+1]
        # concat_rles = F.pad(rles, pad=(1, 0, 0, 0), value=0)

        dec_state, _ = self.decoder(concat_targets, targets_length.add(1))  # [N, L+1, o]

        base_logits, rle_logits = self.joint(enc_state, dec_state)  # [N,L,L+1,vocab_size]

        # print("logits size:", logits.size())

        base_logits = F.log_softmax(base_logits, dim=-1)
        rle_logits = F.log_softmax(rle_logits, dim=-1)

        base_loss = rnnt_loss(base_logits, targets.int(), inputs_length.int(), targets_length.int(), reduction="mean")
        rle_loss = rnnt_loss(rle_logits, rles.int(), inputs_length.int(), targets_length.int(), reduction="mean")
        loss = base_loss + 0.1*rle_loss

        return loss

    # def recognize(self, inputs, inputs_length):
    #
    #     batch_size = inputs.size(0)
    #
    #     enc_states, _ = self.encoder(inputs, inputs_length)  # [N,L,o]
    #
    #     zero_token = torch.LongTensor([[0]])  # [1,1]
    #     if inputs.is_cuda:
    #         zero_token = zero_token.cuda()
    #
    #     def decode(enc_state, lengths):
    #         # enc_state: [L, c]
    #         # length: L
    #         token_list = []
    #
    #         dec_state, hidden = self.decoder(zero_token)  # [1,1,o], [1,1,256],[1,1,256]
    #         print("A:", hidden[0].shape, hidden[1].shape)
    #
    #         for t in range(lengths):  # 限定了预测的最大长度为length，而不是根据特殊字符自动终止的
    #             # while True:
    #             logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))  # [vocab_size]
    #             out = F.softmax(logits, dim=0).detach()  # [vocab_size]
    #             pred = torch.argmax(out, dim=0)  # [1]
    #             pred = int(pred.item())
    #
    #             if pred != 0:
    #                 token_list.append(pred)
    #                 token = torch.LongTensor([[pred]])
    #
    #                 if enc_state.is_cuda:
    #                     token = token.cuda()
    #                 print("B:", token.shape, hidden[0].shape, hidden[1].shape)
    #                 dec_state, hidden = self.decoder(token, hidden=hidden)  #[1, 1], [1, 1, 256], [1, 1, 256]
    #             else:
    #                 break
    #         return token_list
    #
    #     results = []
    #     for i in range(batch_size):
    #         decoded_seq = decode(enc_states[i], inputs_length[i])
    #         results.append(decoded_seq)
    #
    #     return results

    # def recognize(self, inputs, inputs_length):
    #     batch_size = inputs.size(0)
    #     enc_states, _ = self.encoder(inputs, inputs_length)  # [N,L,o]
    #     # print("enc_states shape:", enc_states.shape)
    #     zero_token = torch.zeros(batch_size, 1).long()  # [1,1]
    #     if inputs.is_cuda:
    #         zero_token = zero_token.cuda()
    #
    #     def decode(enc_state, lengths):
    #         # enc_state: [B, 1, c]
    #         # length: [B,]
    #         token_list = []
    #         for _ in range(batch_size):
    #             token_list.append([])
    #
    #         dec_state, hidden = self.decoder(zero_token)  # [B,1,o]
    #         # print("A:", hidden[0].shape, hidden[1].shape)
    #
    #         non_end_set = set()
    #         for id in range(batch_size):
    #             non_end_set.add(id)
    #         for t in range(lengths):  # 限定了预测的最大长度为length，而不是根据特殊字符自动终止的
    #             # while True:
    #             logits = self.joint(enc_state[:, t, :], dec_state.squeeze(1))  # [B,vocab_size]
    #             out = F.softmax(logits, dim=1).detach()  # [B,vocab_size]
    #             pred = torch.argmax(out, dim=1)  # [B,1]
    #             # print(pred.shape)
    #             # print(pred)
    #             non_end = 0
    #             for id in copy.copy(non_end_set):
    #                 if pred[id] != 0:
    #                     token_list[id].append(int(pred[id].item()))
    #                     non_end += 1
    #                 else:
    #                     non_end_set.remove(id)
    #             if len(non_end_set) > 0:
    #                 token = pred.unsqueeze(1)
    #                 # print("AAA")
    #                 # print("B:", token.shape, hidden[0].shape, hidden[1].shape)
    #                 dec_state, hidden = self.decoder(token, hidden=hidden)  #[32]
    #                 # print("BBB")
    #             else:
    #                 break
    #         return token_list
    #
    #     max_length = int(torch.max(inputs_length).item())
    #     # print(max_length)
    #     results = decode(enc_states, max_length)
    #     return results

    def recognize(self, inputs, inputs_length):
        batch_size = inputs.size(0)
        enc_states, _ = self.encoder(inputs, inputs_length)  # [N,L,o]
        # print("enc_states shape:", enc_states.shape)
        zero_token = torch.zeros(batch_size, 1).long()  # [1,1]
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            # enc_state: [B, 1, c]
            # length: [B,]
            token_list = []
            rles_list = []
            for _ in range(batch_size):
                token_list.append([])
                rles_list.append([])

            dec_state, hidden = self.decoder(zero_token)  # [B,1,o]
            # print("A:", hidden[0].shape, hidden[1].shape)

            non_end_set = set()
            for id in range(batch_size):
                non_end_set.add(id)
            for t in range(lengths):  # 限定了预测的最大长度为length，而不是根据特殊字符自动终止的
                # while True:
                base_logits, rle_logits = self.joint(enc_state[:, t, :], dec_state.squeeze(1))  # [B,vocab_size]
                base_out = F.softmax(base_logits, dim=1).detach()  # [B,vocab_size]
                rle_out = F.softmax(rle_logits, dim=1).detach()  # [B,vocab_size]
                pred = torch.argmax(base_out, dim=1)  # [B,1]
                rle_pred = torch.argmax(rle_out, dim=1)  # [B,1]
                # print(pred.shape)
                # print(pred)
                for id in range(batch_size):
                    if pred[id] != 0:
                        token_list[id].append(int(pred[id].item()))
                        rles_list[id].append(int(rle_pred[id].item()))

                token = pred.unsqueeze(1)
                # print("AAA")
                # print("B:", token.shape, hidden[0].shape, hidden[1].shape)
                dec_state, hidden = self.decoder(token, hidden=hidden)  # [32]
            return token_list, rles_list

        max_length = int(torch.max(inputs_length).item())
        # print(max_length)
        results, rle_results = decode(enc_states, max_length)
        return results, rle_results
