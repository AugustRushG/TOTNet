from .wasb import build_wasb
from .tracknet import build_TrackerNet
from .tracknet import build_TrackNetV2
from .TrackNetV4 import TrackNetV4
from .TOTNet import build_motion_model_light
from .TOTNet_OF import build_motion_model_light_opticalflow
from .monoTrack import build_monoTrack
from .TTNet import build_TTNet
from .convlstm import ConvLSTMModel
import json


class Model_Loader:
    def __init__(self, configs):
        self.configs = configs
        self.num_frames = configs.num_frames

    def load_model(self):
        if self.configs.model_choice == 'wasb':
            print("Building WASB model...")
            model = build_wasb(self.configs)
        elif self.configs.model_choice == 'tracknet':
            print("Building TrackNet model...")
            model = build_TrackerNet(self.configs)
        elif self.configs.model_choice == 'tracknetv2':
            print("Building TrackNetV2 model...")
            model = build_TrackNetV2(self.configs)
        elif self.configs.model_choice == 'tracknetv4':
            print("Building TrackNetV4 model...")
            model = TrackNetV4(in_channels=self.num_frames*3, out_channels=1)
        elif self.configs.model_choice == 'TOTNet':
            print("Building Motion Light model...")
            model = build_motion_model_light(self.configs)
        elif self.configs.model_choice == 'TOTNet_OF':
            print("Building Motion Light Optical Flow model...")
            model = build_motion_model_light_opticalflow(self.configs)
        elif self.configs.model_choice == 'monoTrack':
            print("Building MonoTrack")
            model = build_monoTrack(self.configs)
        elif self.configs.model_choice == 'convlstm':
            print("Building ConvLSTM model")
            model = ConvLSTMModel(input_dim=3,
                                  hidden_dim=32,
                                  kernel_size=(3,3),
                                  num_layers=4,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)
        elif self.configs.model_choice == 'TTNet':
            print("Building TTNet")
            model = build_TTNet(self.configs)
        else:
            raise ValueError(f"Unknown model choice: {self.configs.model_choice}")

        return model

    def _load_default_model(self):
        # Logic to load the default model
        pass

    def _load_custom_model(self):
        # Logic to load a custom model
        pass



