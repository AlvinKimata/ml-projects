import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.rawnet import SincConv, Residual_block



class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        model = torchvision.models.resnet18(pretrained=False)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class RawNet(nn.Module):
    def __init__(self, args):
        super(RawNet, self).__init__()

        
        self.device=args.device
        self.filts = [20, [20, 20], [20, 128], [128, 128]]

        self.Sinc_conv=SincConv(device=self.device,
			out_channels = self.filts[0],
			kernel_size = 1024,
            in_channels = args.in_channels)
        
        self.first_bn = nn.BatchNorm1d(num_features = self.filts[0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts = self.filts[1], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = self.filts[1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = self.filts[2]))
        self.filts[2][0] = self.filts[2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts = self.filts[2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = self.filts[2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = self.filts[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features = self.filts[1][-1],
            l_out_features = self.filts[1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features = self.filts[1][-1],
            l_out_features = self.filts[1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features = self.filts[2][-1],
            l_out_features = self.filts[2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features = self.filts[2][-1],
            l_out_features = self.filts[2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features = self.filts[2][-1],
            l_out_features = self.filts[2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features = self.filts[2][-1],
            l_out_features = self.filts[2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features = self.filts[2][-1])
        self.gru = nn.GRU(input_size = self.filts[2][-1],
			hidden_size = args.gru_node,
			num_layers = args.nb_gru_layer,
			batch_first = True)
			
       
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, y = None):
        
        
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x=x.view(nb_samp,1,len_seq)
        
        x = self.Sinc_conv(x)    
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x =  self.selu(x)
        
        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1) # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)
        

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1) # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1 # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1) # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2 # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1) # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3 # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1) # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4 # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1) # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5 # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        output = x[:,-1,:]
      
        return output
        
        

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []
        
        l_fc.append(nn.Linear(in_features = in_features,
			        out_features = l_out_features))

        

        return nn.Sequential(*l_fc)


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

