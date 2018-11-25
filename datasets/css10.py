from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os

from util import audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the css10 dataset from a given input path into a given output directory.'''
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    # Read the transcript file
    with open(os.path.join(in_dir, 'transcript.txt'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            path = os.path.join(in_dir, parts[0])
            text = parts[1]
            futures.append(executor.submit(partial(_process_utterance, out_dir, parts[0].split('/')[1], path, text)))

    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, prompt_id, wav_path, text):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'css10-spec-%s.npy' % prompt_id
    mel_filename = 'css10css10-mel-%s.npy' % prompt_id
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T,  allow_pickle=False)

    # Return a tuple describing this training example:
    n_frames = spectrogram.shape[1]
    return (spectrogram_filename, mel_filename, n_frames, text)
