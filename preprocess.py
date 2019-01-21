import numpy as np
import librosa
from glob import glob

def read_audio(f):
    y, _ = librosa.core.load(f, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    S = np.transpose(np.log(1+10000*S))
    #print('S size', S.shape)
    return S

    #np.save('model_test/avg.npy', avgv)
#np.save('model_test/std.npy', stdv)

def cal(file_path):
    fs = sorted(glob(file_path+'/*.wav'))
    target_fea = []
    for f in fs:
        S = read_audio(f)
        for i in range(S.shape[0]):
            target_fea.append(S[i])
    target_fea = np.array(target_fea)
    print('target_fea size', target_fea.shape)
    avgv = np.sum(target_fea, axis=0)/target_fea.shape[0]
    stdv = np.std(target_fea, axis=0)
    np.save('model_test/avg.npy', avgv)
    np.save('model_test/std.npy', stdv)




    


