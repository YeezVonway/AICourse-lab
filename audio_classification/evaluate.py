import torch
import numpy as np
import librosa

import audio_classification.CONFIG as CONFIG
import audio_classification.models as models
import audio_classification.utils as utils


def get_evalModel(cfg=CONFIG.GLOBAL):
    
    return models.get_model(cfg.FILE.EVAL_MODEL, cfg)


def eval_file(
    model,
    audioFile,
    cfg=CONFIG.GLOBAL
):

    arr, _ = librosa.load(audioFile, sr=cfg.DATA.SR)

    arr = utils.get_frames(
        arr,
        cfg.DATA.FRAME_SIZE,
        cfg.DATA.NUM_FRAMES)
    arr = np.expand_dims(arr, 0)
    x = torch.tensor(
        arr, dtype=torch.float32,
        device=torch.device(cfg.DEVICE)
        )
    y: torch.Tensor = model(x)

    return y.squeeze(dim=0).cpu().detach().numpy()

    