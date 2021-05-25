from .resnet_encoder import resnet18, resnet34
import torch
import torch.nn as nn
import os
import utils



# Citation: Code borrowed from Pytorch. Minor edits by me.
class Model(nn.Module):
    
    def __init__(self, opt, model_dir):
        super(Model, self).__init__()
        self.encoder_frozen = opt.freeze_encoder
        self.pred_mode = opt.prediction_mode
        self.N_history = opt.N_history
        self.title = self.get_title(opt)
        self.model_dir = model_dir

        if opt.encoder == "resnet18":
            self.encoder = resnet18(pretrained=opt.download_encoder)
            self.input_dims = (self.N_history, 3, 96, 96)
        elif opt.encoder == "resnet34":
            self.encoder = resnet34(pretrained=opt.download_encoder)
            self.input_dims = (self.N_history, 3, 96, 96)
        else:
            raise ValueError("Invalid Encoder")


        if self.pred_mode == "regression":
            N_out = 3
        elif self.pred_mode == "classification":
            N_out = 5
        else:
            raise ValueError("Invalid prediction mode")
        self.decoder = self.make_decoder(opt.num_decoder_layers-1, self.encoder.N_out*opt.N_history, 64, N_out)

        # Use to store history
        self.input_buffer = torch.zeros(self.input_dims)

    def make_decoder(self, num_hidden_layers, N_in, N_hidden, N_out):
        layers = []
        current_N = N_in
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_N, N_hidden))
            layers.append(nn.ReLU())
            current_N = N_hidden
            layers.append(nn.BatchNorm1d(current_N))
        layers.append(nn.Linear(current_N, N_out))

        return nn.Sequential(*layers)

    def get_title(self, opt):
        t = f"Encoder={opt.encoder}-freeze_encoder={self.encoder_frozen}-Decoder={opt.num_decoder_layers}-N_history={opt.N_history}-mode={self.pred_mode}"
        if opt.num_train > 0:
            t = f"num_train={opt.num_train}-" + t
        return t

    def get_file_name(self):
        return f"{self.title}.model"

    def load(self, file_name=None):
        if file_name is None:
            file_name = self.get_file_name()
        path = os.path.join(self.model_dir, file_name)
        self.load_state_dict(torch.load(path, map_location=utils.get_device_name()))
        print(f"Loaded model state dict from {path}")

    def save(self, file_name=None):
        if file_name is None:
            file_name = self.get_file_name()
        path = os.path.join(self.model_dir, file_name)
        torch.save(self.state_dict(), path, _use_new_zipfile_serialization=False)
        print(f"Saved model state dict to {path}")

    def to(self, *args, **kwargs):
        new_self = super().to(*args, **kwargs)
        new_self.input_buffer = new_self.input_buffer.to(*args, **kwargs)
        return new_self

    def begin_episode(self):
        self.input_buffer[:, :, :, :] = 0

    # Input Dimensions [batch_size, N_history, n_channels, spatial]
    def forward(self, X):
        with torch.set_grad_enabled(not self.encoder_frozen):
            X_shape = X.size()
            assert self.N_history == X_shape[1], f"History Dimension does not match. Require {self.N_history}"

            X = X.view(X_shape[0]*X_shape[1], X_shape[2], X_shape[3], X_shape[4])
            encoding = self.encoder(X)
            encoding = encoding.view((X_shape[0], self.N_history*self.encoder.N_out))
        output = self.decoder(encoding)
        return output

    # Input dimensions: [n_channels, spatial]
    def infer_action(self, X):
        with torch.no_grad():
            self.input_buffer[1:, :, :, :] = self.input_buffer[:-1, :, :, :].clone()
            self.input_buffer[0, :, :, :] = X

            encoding = self.encoder(self.input_buffer)
            encoding = encoding.view((1, self.N_history*self.encoder.N_out))
            output = self.decoder(encoding)
            output = output.squeeze(0).cpu()
            if self.pred_mode == 'regression':
                return output
            elif self.pred_mode == 'classification':
                label = torch.argmax(output)
                mapping = [[0,0,0], [-1, 0, 0], [1, 0, 0], [0, 0, 0.2], [0, 1, 0]]
                return torch.FloatTensor(mapping[label])
            else:
                raise ValueError("Invalid Prediction Mode")

