import pydub
import numpy as np
from python_speech_features import mfcc
from pydub.silence import split_on_silence
import os
from tqdm import tqdm
import pickle
def get_mfcc_for_sentence(filename):
    x = pydub.AudioSegment.from_file(filename)
    # print(x.frame_rate)
    mfcc_list = []
    audio_chunks = split_on_silence(x, min_silence_len=20, silence_thresh=-33)
    for i, chunk in enumerate(audio_chunks):
        # print(chunk.frame_rate)
        # out_file = "split_{0}.wav".format(i)
        # print("exporting", out_file)
        np_chunk = np.frombuffer(chunk.get_array_of_samples(), dtype=np.int16)
        # print(np_chunk.shape)
        mfcc_feature = mfcc(np_chunk, samplerate=chunk.frame_rate, winlen=0.01)
        print(mfcc_feature.shape)
        mfcc_list.append(mfcc_feature)
        # print(mfcc_feature)
    return mfcc_list
    # samples = chunk.get_array_of_samples()
    # samples = np.array(samples)
    # samples = samples.reshape(chunk.channels, -1, order='F')
    # print(samples)
    # chunk.export(out_file, format="wav")

def process_sentence_file(filename):
    sentence_file = open(filename, "r")
    lines = sentence_file.readlines()
    results = []
    for l in lines:
        spl_l = l.strip("\n").split()
        results.append(spl_l)
    return results

if __name__ == '__main__':
    first_level_dirs = []
    second_level_dirs = []
    training_data = {}
    filename_2_words = {}
    word_2_mfcc = {}
    print(os.listdir("data"))
    for d in os.listdir("data"):
        current_dir = os.path.join("data",d)
        first_level_dirs.append(current_dir)
        for d_second in os.listdir(current_dir):
            second_level_dirs.append(os.path.join(current_dir,d_second))
    print(first_level_dirs)
    print(second_level_dirs)
    # get sentences
    for dir in tqdm(second_level_dirs):
        for file in os.listdir(dir):
            if file.endswith(".txt"):
                abs_file = os.path.join(dir,file)
                word_list = process_sentence_file(abs_file)
                for line in word_list:
                    abs_file = os.path.join(dir,line[0] + ".flac")
                    filename_2_words[abs_file] = line[1:]
                    # print(filename_2_words[abs_file])

    # get mfccs
    count = 0
    for dir in tqdm(second_level_dirs):
        for file in os.listdir(dir):
            if file.endswith(".flac"):
                abs_file = os.path.join(dir,file)
                training_data[abs_file] = get_mfcc_for_sentence(abs_file)
                if len(training_data[abs_file]) == len(filename_2_words[abs_file]):
                    print("match!!")
                else:
                    print("length miss match at:", abs_file, "length diff:",len(training_data[abs_file]) - len(filename_2_words[abs_file]))

    print(len(training_data))
    with open('train_data.pkl',"wb") as train_file:
        pickle.dump(training_data,train_file)
    # with open('train_data.pkl',"rb") as train_file:
    #     training_data = pickle.load(train_file)
    # for i in training_data.keys():
    #     print(training_data[i][0].shape)
