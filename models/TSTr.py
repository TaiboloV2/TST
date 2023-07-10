import torch, torchaudio, math
import torch.nn as nn
import torch.nn.functional as F
from custom_layers.NormMinMax import NormMinMax


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
        self.d_model = 8
        self.d_ff = 2 * self.d_model
        self.activation = torch.nn.functional.relu
        self.n_head = 2
        self.n_encoder_layers = 6
        assert self.d_model % self.n_head == False, "d_model must be a multiple of n_head!"
        self.encoder_dropout = 0

        # time - 1 == true, 0 == false
        self.use_time = 1
        self.use_rms = 0
        self.embedding_bias = 0
        self.positional_encoding = 1
        
        # training
        self.batch_size = 32
        self.learning_rate = 1e-4

        # hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(self.device == "cpu"): torch.set_num_threads(8)

        torch.set_flush_denormal(True)


# ---------------------------------------------------------------------------------------------------------------------
class STFTTr(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()

        # config
        self.model_config = model_config

        # norm
        self.norm = NormMinMax()

        # STFT
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=127, window_fn=torch.hann_window, win_length=127, hop_length=40, normalized=True)
        # x = torchaudio.transforms.MelSpectrogram(sample_rate=25600, normalized=True)(x)

        # patching
        self.patch_sidelength = int(torch.sqrt(torch.tensor([model_config.d_patches])))
        self.patching_layer = torch.nn.Unfold(kernel_size=(self.patch_sidelength,self.patch_sidelength), stride=(self.patch_sidelength,self.patch_sidelength))
        self.patch_dropout = nn.Dropout2d(model_config.patch_dropout)

        # linear projection
        self.lp_layer = nn.Linear(model_config.d_patches, model_config.d_model, bias=model_config.embedding_bias)
        self.time_embedding = nn.Linear(1, model_config.d_model, bias=model_config.embedding_bias)

        # # class token
        self.class_token = nn.Parameter(torch.randn((1,1,model_config.d_model)), requires_grad=True)

        # learnable PE layer
        self.pe_layer = torch.nn.parameter.Parameter(torch.empty((model_config.n_patches + self.model_config.use_time + self.model_config.use_rms, model_config.d_model)).uniform_(-0.02, 0.02), requires_grad=True)
        self.pe_dropout = nn.Dropout(model_config.pe_dropout)

        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(model_config.d_model, model_config.n_head, model_config.d_ff, model_config.encoder_dropout, model_config.activation, norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, model_config.n_encoder_layers)

        # mlp head
        self.ln1 = nn.Linear((model_config.n_patches + self.model_config.use_time + self.model_config.use_rms) * model_config.d_model, 512)
        self.ln2 = nn.Linear(512, 128)
        # self.ln3 = nn.Linear(128, 32)
        self.ln4 = nn.Linear(128, 1)


    def forward(self, x):
        # split input
        time = x[:,-1].contiguous()
        x = x[:,:-1].contiguous()

        # # norm time series
        # x = self.norm(x)

        # STFT - 64x64
        x = self.spectrogram(x)
        x = torch.absolute(x) # magnitude
        x = 20 * torch.log10(x) # to dB
        # x = self.norm(x)
        x /= 10

        # x = torch.transpose(x, 1, 2)

        if self.model_config.visualize_patching:
            import matplotlib.pyplot as plt
            plt.imshow(x[0], cmap="jet", vmin=x[0].min(), vmax=x[0].max())
            plt.show()

        # patching - 64x64
        x = x.unsqueeze(1)
        x = self.patching_layer(x)
        x = x.permute(0,2,1)

        if self.model_config.visualize_patching:
            fig, ax = plt.subplots(int(math.sqrt(self.model_config.n_patches)), int(math.sqrt(self.model_config.n_patches)))
            k = 0
            for row in ax:
                for col in row:
                    col.imshow(torch.reshape(x[0][k], (self.patch_sidelength,self.patch_sidelength)), cmap="jet", vmin=x[0].min(), vmax=x[0].max())
                    col.set_axis_off()
                    k += 1
            fig.set_size_inches(w=4, h=4)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()

        # project d_patches to d_model
        x = self.lp_layer(x)

        # add rms token
        if self.model_config.use_rms:
            rms = self.rms_embedding(rms.unsqueeze(1))
            x = torch.cat((rms.unsqueeze(1), x), dim=1)

        # add time token
        if self.model_config.use_time:
            time[time == -10000] = 0
            time /= 100
            time = self.time_embedding(time.unsqueeze(1))
            x = torch.cat((time.unsqueeze(1), x), dim=1)

        # add class token
        # x = torch.cat((self.class_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # add PE
        if(self.model_config.positional_encoding):
            x = x + self.pe_layer
            x = self.pe_dropout(x)

        # encoder pass
        x = self.encoder(x)
        # x = x[:,0]

        # mlp head
        x = torch.flatten(x, 1)

        x = self.ln1(x)
        x = F.relu(x)
        # x = nn.LayerNorm(x.shape[1], elementwise_affine=True)(x)

        x = self.ln2(x)
        x = F.relu(x)

        # x = self.ln3(x)
        # x = F.relu(x)
        
        x = self.ln4(x)
        # x = F.sigmoid(x)

        # x = F.relu(x)

        return x