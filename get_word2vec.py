import numpy as np
import librosa
import os
from tqdm import tqdm
import pickle
import gensim
from gensim.models import word2vec


def process_sentence_file(filename):
    sentence_file = open(filename, "r")
    lines = sentence_file.readlines()
    results = []
    for l in lines:
        spl_l = l.strip("\n").split(" ")[1:]
        spl_l = [s.lower() for s in spl_l]
        results.append(spl_l)
    return results


def split_on_silences(audio_fpath, words, end_times):
    # Load the audio waveform
    sample_rate = 16000  # Sampling rate of LibriSpeech
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
    sentences_list = []
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
                sentence_mfcc = []
                sentence_words = []
                for i, w in enumerate(words):
                    if w.isspace() or not w:
                        continue
                    else:
                        sentence_words.append(w.lower())
                sentences_list.append(sentence_words)
    model = word2vec.Word2Vec(sentences_list, sg=1, size=100, window=3, min_count=1)  # sg=1 表示skip-gram模型
    # print(model['keep'])
    # print(model.most_similar(['love']))
    model.wv.save_word2vec_format('word2vec_first_50.txt', binary=False)
    with open("sentences_list.pkl",'wb') as  file:
        pickle.dump(sentences_list,file)