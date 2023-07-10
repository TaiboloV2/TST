import torch, torchaudio, math
import torch.nn as nn
import torch.nn.functional as F
from custom_layers.NormMinMax import NormMinMax
from custom_layers.STFT_layer import STFT_layer


class ModelConfig():
    def __init__(self):

        # patching
        self.n_patches = 64
        self.d_patches = 64
        self.patch_dropout = 0
        self.visualize_patching = False

        # positional encoding
        self.pe_dropout = 0

        # transformer
        self.d_model = 64
        self.d_ff = 4 * self.d_model
        self.activation = torch.nn.functional.relu
        self.n_head = 1
        self.n_encoder_layers = 1
        assert self.d_model % self.n_head == False, "d_model must be a multiple of n_head!"
        self.encoder_dropout = 0.1
        self.n_classes = 4

        # time -> 1 == true, 0 == false
        self.use_mic = 0
        self.sensor_fusion = 0
        self.embedding_bias = 0
        self.positional_encoding = 1
        
        # training
        self.batch_size = 8
        self.learning_rate = 1e-4

        # hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(self.device == "cpu"): torch.set_num_threads(8)

        torch.set_flush_denormal(True)

        # telemetry
        if(self.sensor_fusion):
            print("sensor fusion")
        elif(self.use_mic):
            print("microphone")
        else:
            print("accelerometer")


# ---------------------------------------------------------------------------------------------------------------------
class TSTc(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()

        # config
        self.model_config = model_config

        # norm
        self.norm = NormMinMax()

        # STFT
        self.STFT = STFT_layer()

        # patching
        self.patch_sidelength = int(torch.sqrt(torch.tensor([model_config.d_patches])))
        self.patching_layer = torch.nn.Unfold(kernel_size=(self.patch_sidelength,self.patch_sidelength), stride=(self.patch_sidelength,self.patch_sidelength))
        self.patch_dropout = nn.Dropout2d(model_config.patch_dropout)

        # linear projection
        self.lp_layer = nn.Linear(model_config.d_patches, model_config.d_model, bias=model_config.embedding_bias)
        self.time_embedding = nn.Linear(1, model_config.d_model, bias=model_config.embedding_bias)

        # # class token
        # self.class_token = nn.Parameter(torch.randn((1,1,model_config.d_model)), requires_grad=True)

        # learnable PE layer
        self.pe_layer = torch.nn.parameter.Parameter(torch.empty((model_config.n_patches * (1 + model_config.sensor_fusion), model_config.d_model)).uniform_(-0.02, 0.02), requires_grad=True)
        self.pe_dropout = nn.Dropout(model_config.pe_dropout)

        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(model_config.d_model, model_config.n_head, model_config.d_ff, model_config.encoder_dropout, model_config.activation, norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, model_config.n_encoder_layers)

        # mlp head
        self.ln1 = nn.Linear((model_config.n_patches * (1 + model_config.sensor_fusion)) * model_config.d_model, 512)
        # self.ln1 = nn.Linear(model_config.d_model, 256)
        self.ln2 = nn.Linear(512, 128)
        # self.ln3 = nn.Linear(128, 256)
        self.ln4 = nn.Linear(128, model_config.n_classes)


    def forward(self, x):
        # split input
        acc = x[:,:2560]
        mic = x[:,2560:]

        if self.model_config.sensor_fusion:
            acc = self.STFT(acc)
            acc = acc.unsqueeze(1)
            acc = self.patching_layer(acc)
            acc = acc.permute(0,2,1)
            mic = self.STFT(mic)
            mic = mic.unsqueeze(1)
            mic = self.patching_layer(mic)
            mic = mic.permute(0,2,1)

            x = torch.cat([acc, mic], dim=1)

        elif self.model_config.use_mic:
            x = mic
            x = self.STFT(x)
            x = x.unsqueeze(1)
            x = self.patching_layer(x)
            x = x.permute(0,2,1)
        else:
            x = acc
            x = self.STFT(x)
            x = x.unsqueeze(1)
            x = self.patching_layer(x)
            x = x.permute(0,2,1)

        src = x
        # # norm time series
        # x = self.norm(x)

        # project d_patches to d_model
        x = self.lp_layer(x)

        # # add class token
        # x = torch.cat((self.class_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # add PE
        if(self.model_config.positional_encoding):
            x = x + self.pe_layer
            x = self.pe_dropout(x)

        # encoder pass
        x = self.encoder(x)
        plt_attn = False
        if plt_attn:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(int(math.sqrt(self.model_config.n_patches)), int(math.sqrt(self.model_config.n_patches)))
            k = 0
            for row in ax:
                for col in row:
                    col.imshow(torch.reshape(src[0][k].detach(), (self.patch_sidelength,self.patch_sidelength)), cmap="jet", vmin=x[0].min(), vmax=x[0].max())
                    col.set_axis_off()
                    k += 1
            fig.set_size_inches(w=4, h=4)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()

            attn_weights = self.norm(attn_weights)
            src = torch.mul(src, attn_weights[0])
            fig, ax = plt.subplots(int(math.sqrt(self.model_config.n_patches)), int(math.sqrt(self.model_config.n_patches)))
            k = 0
            for row in ax:
                for col in row:
                    col.imshow(torch.reshape(src[0][k].detach(), (self.patch_sidelength,self.patch_sidelength)), cmap="jet", vmin=x[0].min(), vmax=x[0].max())
                    col.set_axis_off()
                    k += 1
            fig.set_size_inches(w=4, h=4)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()

        # x = x[:,0]

        # mlp head
        x = torch.flatten(x, 1)

        x = self.ln1(x)
        x = F.relu(x)

        x = self.ln2(x)
        x = F.relu(x)

        # x = self.ln3(x)
        # x = F.relu(x)
        
        x = self.ln4(x)
        x = F.softmax(x, dim=1)

        return x