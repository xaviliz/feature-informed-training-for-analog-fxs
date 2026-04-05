import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal.windows import tukey
import torch.nn.functional as F
from pathlib import Path
from extract_features import EssentiaExtractor
from utils.dir import dataset_dir


class DataGeneratorPickles(Dataset):
    def __init__(
        self,
        data_dir: Path,
        filename: str,
        mini_batch_size: int,
        batch_size: int,
        set: str,
        model: str,
        feature: str,
        extractor: str,
        predict_feature: bool,
        samplerate: int = 48000,
        stateful: bool = True,
        use_multiband: bool = True,
        lim_for_testing: bool = False,
        extract_in_loading: bool = False,
    ) -> None:
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param mini_batch_size: mini batch size
          :param batch_size: The size of each batch returned by __getitem__
          :param set: dataset split (train/val/test)
          :param model: model reference (for stateful operations)
          :param stateful: whether to use stateful operations
        """

        self.indices2 = None
        self.indices = None
        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.frame_size = mini_batch_size
        self.predict_feature = predict_feature
        self.feature = feature
        self.model = model
        self.stateful = stateful
        self.set = set
        self.samplerate = samplerate
        self.hop_size = mini_batch_size
        self.lim_for_testing = lim_for_testing
        self.extract_in_loading = extract_in_loading

        self.x, self.y, self.z = self.prepareXYZ(data_dir, filename)
        assert self.x.shape[0] % self.batch_size == 0
        self.z = np.repeat(self.z, self.x.shape[1], axis=1)
        self.output_size = self.y.shape[-1]

        if self.extract_in_loading == False:
            if self.feature is not None:
                self.extractorObj = EssentiaExtractor(
                    samplerate=self.samplerate,
                    frame_size=self.frame_size,
                    hop_size=self.hop_size,
                )
                num_audios = self.x.shape[0]
                self.feature_input, self.feature_target = ([] for _ in range(2))
                print(f"Extracting {feature} for {num_audios} audio files...")

                for i in range(self.x.shape[0]):
                    self.feature_input.append(
                        self.extractorObj.process_audio(
                            self.x[i], self.samplerate, feature
                        )[feature].T
                    )  # B, T, 1
                    self.feature_target.append(
                        self.extractorObj.process_audio(
                            self.y[i], self.samplerate, feature
                        )[feature].T
                    )

                self.feature_input = torch.from_numpy(
                    np.array(self.feature_input, dtype=np.float32)
                )
                self.feature_target = torch.from_numpy(
                    np.array(self.feature_target, dtype=np.float32)
                )

                if feature == "envelope":
                    self.feature_input = self.feature_input.reshape(
                        self.feature_input.shape[0], -1
                    )
                    self.feature_target = self.feature_target.reshape(
                        self.feature_target.shape[0], -1
                    )

                # Convert to PyTorch tensors
                self.x = torch.from_numpy(self.x)  # B, T, 1
                self.y = torch.from_numpy(self.y)  # B, T, 1
                self.z = torch.from_numpy(self.z)  # B, T, P

                # if self.encode_input:
                #    self.x = self.ae.encoder(self.x.permute(0, 2, 1).repeat(1, 2, 1)).permute(0, 2, 1)
                #    self.y = self.ae.encoder(self.y.permute(0, 2, 1).repeat(1, 2, 1)).permute(0, 2, 1)

                if len(self.feature_input.shape) > 3:
                    self.feature_input.squeeze(-1)
                    self.feature_target.squeeze(-1)

                if len(self.feature_input.shape) == 2:
                    self.feature_input = self.feature_input.unsqueeze(2)
                    self.feature_target = self.feature_target.unsqueeze(2)

                self.feature_input = self.feature_input.permute(0, 2, 1)
                self.feature_input = F.interpolate(
                    self.feature_input, size=self.x.shape[1], mode="linear"
                )  # B, T, 1
                self.feature_input = self.feature_input.permute(0, 2, 1)
                self.z = torch.cat((self.z, self.feature_input), dim=2)  # B, T, P+1

                if self.predict_feature:
                    self.feature_target = self.feature_target.permute(0, 2, 1)
                    self.feature_target = F.interpolate(
                        self.feature_target, size=self.y.shape[1], mode="linear"
                    )  # B, T, P+1
                    self.feature_target = self.feature_target.permute(0, 2, 1)
                    self.y = torch.cat(
                        (self.y, self.feature_target), dim=2
                    )  # B, T, 1+1
                self.output_size = self.y.shape[-1]
                self.conditioning_dim = self.z.shape[-1]
            else:
                # Convert to PyTorch tensors
                self.x = torch.from_numpy(self.x)  # B, T, 1
                self.y = torch.from_numpy(self.y)  # B, T, 1
                self.z = torch.from_numpy(self.z)  # B, T, P

                self.output_size = self.y.shape[-1]
                self.conditioning_dim = self.z.shape[-1]
        else:
            self.extractorObj = EssentiaExtractor(
                samplerate=self.samplerate,
                frame_size=self.frame_size,
                hop_size=self.hop_size,
            )
            ValueError("EssentiaExtractor() is not available")

            # Convert to PyTorch tensors
            self.x = torch.from_numpy(self.x)  # B, T, 1
            self.y = torch.from_numpy(self.y)  # B, T, 1
            self.z = torch.from_numpy(self.z)  # B, T, P

            self.output_size = self.y.shape[-1]
            self.get_dimensions()

        self.num_audios = self.x.shape[0]
        self.idj = 0
        self.idx = -1

        self.max_1 = (self.x.shape[1] // self.frame_size) - 1
        self.max_2 = self.x.shape[0] // self.batch_size
        self.training_steps = self.max_1 * self.max_2
        self.on_epoch_end()

    def prepareXYZ(self, data_dir: Path, filename: str):  # -> tuple:
        print(f"filename: {filename}")
        file_data = open(data_dir / filename, "rb")
        Z = pickle.load(file_data)
        file_data.close()

        if self.lim_for_testing:
            x = np.array(Z["x"][:2, :48000, None], dtype=np.float32)
            y = np.array(Z["y"][:2, :48000, None], dtype=np.float32)
            z = np.array(Z["z"][:2, :48000], dtype=np.float32)
        else:
            x = np.array(Z["x"][:, :, None], dtype=np.float32)
            y = np.array(Z["y"][:, :, None], dtype=np.float32)
            z = np.array(Z["z"][:, :], dtype=np.float32)

            if filename[:4] == "LA2A":
                z1 = z[:, 0:1] / 10
                z2 = z[:, 1:2]
                z = np.concatenate((z1, z2), axis=-1)

                indices = np.nonzero(z[:, 1] == 0.0)[0]
                z = z[indices, 0:1]
                x = x[indices]
                y = y[indices]

            if filename[:2] == "OD":
                z = z.T
                z1 = z[:, 0:1]
                z2 = z[:, 1:2]
                z = np.concatenate((z1, z2), axis=-1)

                indices = np.nonzero(z[:, 1] == 0.0)[0]
                z = z[indices, 0:1]
                x = x[indices]
                y = y[indices]

        if z.shape[0] < z.shape[1]:
            z = z.T

        # Apply Tukey window
        tukey_window = np.array(
            tukey(x.shape[1], alpha=0.000005), dtype=np.float32
        ).reshape(1, -1, 1)
        x = x * tukey_window
        y = y * tukey_window

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        N = int((x.shape[1]) / self.frame_size)  # how many iterations
        lim = int(N * self.frame_size)  # how many samples
        x = x[:, :lim]
        y = y[:, :lim]

        z = z[:, np.newaxis, :]
        self.device_param = z.shape[-1]
        return x, y, z

    def getXY(self):
        """
        Get all X, Y pairs for the entire dataset
        Returns tensors instead of lists for PyTorch compatibility
        """
        Xs, Ys = [], []
        for idx in range(self.__len__()):
            X, Y = self.__getitem__(idx)
            Xs.append(X)
            Ys.append(Y)

        return torch.stack(Xs), torch.stack(Ys)

    def on_epoch_end(self) -> None:
        """Reset indices at the end of each epoch"""
        self.indices = np.arange(0, self.x.shape[1])
        self.indices2 = np.arange(0, self.x.shape[0])
        self.idj = 0
        self.idx = -1

        # Reset model states if stateful and model has the method
        if self.stateful and self.model is not None:
            if hasattr(self.model, "reset_hidden_states"):
                self.model.reset_hidden_states()

    def get_dimensions(self) -> None:
        """To compute the conditioning ad output dimension in case of feature"""
        feature_input, feature_target = ([] for _ in range(2))
        print(f"Extracting {self.feature} for the one batch of files...")

        feature_input.append(
            self.extractorObj.process_audio(
                self.x[0].numpy(), self.samplerate, self.feature
            )[self.feature].T
        )  # B, T, 1
        feature_target.append(
            self.extractorObj.process_audio(
                self.y[0].numpy(), self.samplerate, self.feature
            )[self.feature].T
        )

        feature_input = torch.from_numpy(np.array(feature_input, dtype=np.float32))
        feature_target = torch.from_numpy(np.array(feature_target, dtype=np.float32))

        if self.feature == "envelope":
            feature_input = feature_input.reshape(feature_input.shape[0], -1)
            feature_target = feature_target.reshape(feature_target.shape[0], -1)

        if len(feature_input.shape) > 3:
            feature_input.squeeze(-1)
            feature_target.squeeze(-1)

        if len(feature_input.shape) == 2:
            feature_input = feature_input.unsqueeze(2)
            feature_target = feature_target.unsqueeze(2)

        self.conditioning_dim = self.z.shape[-1] + feature_input.shape[-1]
        if self.predict_feature:
            self.output_size = self.y.shape[-1] + feature_target.shape[-1]

    def __len__(self) -> int:
        return int(self.training_steps)

    def __getitem__(self, idx):  # -> tuple:
        """
        Get a single batch item
        Returns: (X, Y) where X is input sequences and Y is target sequences
        """
        ############################################################################
        ############################################################################
        if self.feature is not None and self.extract_in_loading:
            feature_input, feature_target = ([] for _ in range(2))
            print(f"Extracting {self.feature} for the current batch of files...")

            for i in range(self.x.shape[0]):
                feature_input.append(
                    self.extractorObj.process_audio(
                        self.x[i].numpy(), self.samplerate, self.feature
                    )[self.feature].T
                )  # B, T, 1
                feature_target.append(
                    self.extractorObj.process_audio(
                        self.y[i].numpy(), self.samplerate, self.feature
                    )[self.feature].T
                )

            feature_input = torch.from_numpy(np.array(feature_input, dtype=np.float32))
            feature_target = torch.from_numpy(
                np.array(feature_target, dtype=np.float32)
            )

            if self.feature == "envelope":
                feature_input = feature_input.reshape(feature_input.shape[0], -1)
                feature_target = feature_target.reshape(feature_target.shape[0], -1)

            if len(feature_input.shape) > 3:
                feature_input.squeeze(-1)
                feature_target.squeeze(-1)

            if len(feature_input.shape) == 2:
                feature_input = feature_input.unsqueeze(2)
                feature_target = feature_target.unsqueeze(2)

            conditioning = torch.cat(
                (self.conditioning, feature_input), dim=2
            )  # B, T, P+1

            if self.predict_feature:
                target = torch.cat((self.target, feature_target), dim=2)  # B, T, 1+1
        ############################################################################
        ############################################################################

        # Initialize batch tensors
        X = torch.zeros((self.batch_size, self.frame_size, 1), dtype=torch.float32)
        Y = torch.zeros(
            (self.batch_size, self.frame_size, self.output_size), dtype=torch.float32
        )
        Z = torch.zeros(
            (self.batch_size, self.frame_size, self.z.shape[-1]), dtype=torch.float32
        )

        if idx == 0:
            self.idj = 0
            self.idx = -1

        # Handle batch transitions
        if idx % self.max_1 - 1 == 0 and idx != 1:
            self.idj += 1
            self.idx = -1
            if self.stateful and self.model is not None:
                if hasattr(self.model, "reset_hidden_states"):
                    self.model.reset_hidden_states()

        self.idx += 1

        # Get the indices of the requested batch
        indices = self.indices[
            self.idx * self.frame_size : (self.idx + 1) * self.frame_size
        ]
        indices2 = self.indices2[
            self.idj * self.batch_size : (self.idj + 1) * self.batch_size
        ]

        c = 0
        for t in range(indices[0], indices[-1] + 1, 1):
            X[:, c, :] = self.x[indices2, t, :]
            Y[:, c, :] = self.y[indices2, t, :]
            Z[:, c, :] = self.z[indices2, t, :]
            c += 1

        return X, Y, Z


if __name__ == "__main__":
    data_dir = dataset_dir / "audio-effects-datasets-vol-1"
    dataset_name = "od300"
    dataset = DataGeneratorPickles(
        data_dir=data_dir,
        filename=dataset_name + "_test.pickle",
        mini_batch_size=512,
        batch_size=1,
        set="train",
        model="lstm",
        feature=None,
        extractor="librosa",
        predict_feature=False,
        stateful=True,
        use_multiband=False,
        lim_for_testing=False,
        extract_in_loading=False,
    )
