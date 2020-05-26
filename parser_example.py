# import sounddevice as sd
import numpy as np
import librosa
import os
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
# This script serves as example to parsing the txt alignments. It will iterate over all 
# utterances of the dataset, split utterances on silences that are longer 0.4s and play them.
# (this does not change your files)

   # Replace with yours

def split_on_silences(audio_fpath, words, end_times):
    # Load the audio waveform
    sample_rate = 16000     # Sampling rate of LibriSpeech 
    wav, _ = librosa.load(audio_fpath, sample_rate)
    
    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == '' and words[-1] == ''
    
    # Break the sentence on pauses that are longer than 0.4s
    mask = [True] * (len(words))
    # mask = (words == '')
    breaks = np.where(mask)[0]
    segment_times = [[end_times[s], start_times[e]] for s, e in zip(breaks, breaks)]
    segment_times = (np.array(segment_times) * sample_rate).astype(np.int)
    wavs = [wav[segment_time[1]:segment_time[0]] for segment_time in segment_times]
    return wavs

if __name__ == '__main__':
    file_2_mfccs = {}
    word_2_mfcc = {}
    set_dir = "data_last_50"
    # Select speakers
    for speaker_id in tqdm(os.listdir(set_dir)):
        speaker_dir = os.path.join(set_dir, speaker_id)

        # Select books
        for book_id in os.listdir(speaker_dir):
            book_dir = os.path.join(speaker_dir, book_id)

            # Get the alignment file
            alignment_fpath = os.path.join(book_dir, "%s-%s.alignment.txt" %
                                           (speaker_id, book_id))
            if not os.path.exists(alignment_fpath):
                raise Exception("Alignment file not found. Did you download and merge the txt "
                                "alignments with your LibriSpeech dataset?")

            # Parse each utterance present in the file
            alignment_file = open(alignment_fpath, "r")
            for line in alignment_file:
                # Retrieve the utterance id, the words as a list and the end_times as a list
                utterance_id, words, end_times = line.strip().split(' ')

                words = words.replace('\"', '').split(',')
                end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
                audio_fpath = os.path.join(book_dir, utterance_id + '.flac')

                # Split utterances on silences
                wavs = split_on_silences(audio_fpath, words, end_times)
                sentence_mfcc = []
                assert len(wavs) == len(words)
                for i,w in enumerate(words):
                    if w.isspace() or not w:
                        continue
                    else:
                        mfcc_feature = librosa.feature.mfcc(wavs[i], sr=16000, n_mfcc=13)
                        sentence_mfcc.append(mfcc_feature.T)
                        if w not in word_2_mfcc.keys():
                            word_2_mfcc[w] = mfcc_feature.T
                file_2_mfccs[audio_fpath] = sentence_mfcc
            alignment_file.close()
    print("size of recorded words:" ,len(word_2_mfcc.keys()))
    with open('train_data_last_50.pkl',"wb") as train_file:
        pickle.dump(file_2_mfccs,train_file)
    with open("word_2_mfcc_last_50.pkl","wb") as wordmfcc:
        pickle.dump(word_2_mfcc, wordmfcc)
