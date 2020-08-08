import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size, conv_dim=1, padding=0, bias=True, activation=False, dilation=1):
        super(DepthWiseSeparableConv, self).__init__()
        self.activation = activation
        if conv_dim == 1:
            self.conv = nn.Conv1d(in_channels=input_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                  bias=bias, dilation=dilation)
        if activation:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(self.conv.weight)

    def forward(self, input):
        input = input.transpose(1, 2)  # cause the input format of CNN is (batch_size, dim ,len)
        out = self.conv(input)
        out = out.transpose(1, 2)
        if self.activation:
            return F.relu(out)
        else:
            return out


def GLU(input):
    output_dim = input.shape[2] // 2
    a, b = torch.split(input, output_dim, dim=2)
    return a * torch.sigmoid(b)


class Similarity(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(Similarity, self).__init__()
        self.dropout = dropout
        self.ff1 = nn.Linear(10, 10)
        self.ff2 = nn.Linear(10, 1)

    def forward(self, article: torch.Tensor, title: torch.Tensor, article_mask: torch.Tensor, title_mask: torch.Tensor):
        article_len = article.shape[1]
        title_len = title.shape[1]
        c = article.unsqueeze(dim=2)
        c = c.repeat([1, 1, title_len, 1])
        q = title.unsqueeze(dim=1)
        q = q.repeat([1, article_len, 1, 1])

        s = q - c
        s = torch.sum(s * s, dim=3)
        s = torch.exp(-1 * s)

        article_mask = article_mask.unsqueeze(dim=2)
        article_mask = article_mask.repeat([1, 1, title_len])
        title_mask = title_mask.unsqueeze(dim=1)
        title_mask = title_mask.repeat([1, article_len, 1])

        s = s * article_mask * title_mask

        row_max, _ = torch.max(s, dim=2)
        row_max, _ = row_max.topk(10, dim=1, largest=True)

        row_max = F.relu(self.ff1(row_max))
        row_max = self.ff2(row_max)
        out = row_max
        out = out.squeeze()
        return out


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthWiseSeparableConv(input_dim=input_dim, out_dim=input_dim * 2, kernel_size=kernel_size,
                                            padding=kernel_size // 2 * dilation, dilation=dilation)

    def forward(self, input):
        out = self.conv1(input)
        out = GLU(out)
        out = input + out
        return out


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, kernel_size, conv_num, exp_num=5, refine_num=5, dropout=0.1):
        super(ConvEncoder, self).__init__()
        self.dropout = dropout
        self.exp_conv = nn.Sequential()
        dilation = 1
        for i in range(conv_num):
            self.exp_conv.add_module(str(i),
                                     ResidualBlock(input_dim=input_dim, kernel_size=kernel_size, dilation=dilation))
            if i < exp_num:
                dilation *= 2
        self.refine = nn.Sequential()
        for i in range(refine_num):
            self.refine.add_module(str(i), ResidualBlock(input_dim=input_dim, kernel_size=kernel_size, dilation=1))

    def forward(self, input):
        out = self.exp_conv(input)
        out = self.refine(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class FastRerank(nn.Module):
    def __init__(self, word_dim, word_mat, kernel_size, encoder_block_num, dropout=0.1):
        super(FastRerank, self).__init__()
        self.word_emb = nn.Embedding(word_mat.shape[0], word_dim)
        self.conv_encoder = ConvEncoder(word_dim, kernel_size, encoder_block_num, dropout=dropout)
        self.similarity = Similarity(word_dim, dropout)

    def forward(self, article_word, title_word, article_mask, title_mask):
        article_word = self.word_emb(article_word)
        title_word = self.word_emb(title_word)
        article = article_word
        title = title_word
        article = self.conv_encoder(article)
        title = self.conv_encoder(title)
        score = self.similarity(article, title, article_mask, title_mask)
        return score