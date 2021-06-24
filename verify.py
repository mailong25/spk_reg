import pytorch_lightning as pl
import uuid
import pickle, json
import shutil
import torch, os
from omegaconf import OmegaConf
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
import numpy as np

import wave
def get_length(audio):
    info = wave.open(audio, 'rb')
    return round(float(info.getnframes()) / info.getframerate(),2)

def cos_sim(X,Y):
    score = (X @ Y.T) / (((X @ X.T) * (Y @ Y.T)) ** 0.5)
    score = (score + 1) / 2
    return score

torch.set_grad_enabled(False)

class SpeakerVerifier:
    
    def __init__(self, model_path, database_dir):
        '''
        model_path: path to verification model
        database_dir: directory to store user embedding
        user embedding will be save as $user_id.npy in database_dir
        '''
        
        self.speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(restore_path=model_path)
        self.trainer = pl.Trainer(gpus=0, accelerator=None)
        self.database = os.path.abspath(database_dir)
    
    def extract_embedding(self,wavs):
        '''
        Input: wavs: paths to wavs
        Output: numpy array (len(wavs), 256)
        Extract embeddings from wav files. Each wav file will produce one embedding.
        Embedding is a 256 dim numpy array
        '''
        
        temp_dir = uuid.uuid4().hex
        os.mkdir(temp_dir)
        temp_dir = os.path.abspath(temp_dir)
        manifest_filepath = os.path.join(temp_dir,'manifest.json')
        embedding_path = os.path.join(temp_dir,'embeddings','manifest_embeddings.pkl')
        wavs = [os.path.abspath(p) for p in wavs]
        
        manifest = []
        for i in range(0,len(wavs)):
            if wavs[i] in wavs[:i]:
                new_name = os.path.join(temp_dir, uuid.uuid4().hex + '.wav')
                shutil.copy(wavs[i],new_name)
                wavs[i] = new_name
            
            manifest.append(json.dumps({"audio_filepath": wavs[i], "duration": get_length(wavs[i]), "label": 'x'}))
            
        with open(manifest_filepath,'w') as f:
            f.write('\n'.join(manifest))
        
        test_config = OmegaConf.create(
            dict(
                manifest_filepath=manifest_filepath,
                sample_rate=16000, labels=None,
                batch_size=1, shuffle=False,
                time_length=8, embedding_dir=temp_dir,
            ))
        
        self.speaker_model.setup_test_data(test_config)
        self.trainer.test(self.speaker_model)
        
        data = pickle.load(open(embedding_path,'rb'))
        
        embeddings = []
        for path in wavs:
            for key in data:
                if key in '@'.join(path.split('/')):
                    embeddings.append(data[key])
                    break
        
        shutil.rmtree(temp_dir)
        return np.vstack(embeddings)

    def enrol(self, wavs, spk_id):
        '''
        Register user voice to the system
        Input: 
        + wavs: audio files spoken by the user. At least 3 audios (4->8s long) is recommended.
        + spk_id: user id
        
        Output: 
        + True -> Enrol user sucessful
        + False -> user id is already exists
        '''
        
        spk_emd = os.path.join(self.database, spk_id + '.npy')
        if os.path.exists(spk_emd):
            # User already exists
            return False
        
        embeddings = self.extract_embedding(wavs)
        np.save(spk_emd, embeddings)
        return True
        
    def verify(self, wav, spk_id, threshold = 0.7):
        '''
        Input: 
        + wav: audio need to be verify
        + spk_id: user id need to be verify
        + threshold: range from 0 -> 1. This is rejection threshold. 
          The higher the value, the higher chance that the audio will be rejected (classified as different speaker) 
        
        Output: 
        + True -> same speaker
        + False -> different speaker
        '''
        
        spk_emd = os.path.join(self.database, spk_id + '.npy')
        if not os.path.exists(spk_emd):
            return False
        
        spk_emd = np.load(spk_emd)
        wav_emb = self.extract_embedding([wav])[0]
        
        similarity = [cos_sim(wav_emb,e) for e in spk_emd]
        similarity = np.mean(similarity)
        
        if similarity > threshold:
            return True
        else:
            return False
    
    def remove(self,spk_id):
        '''
        Remove spk_id from the database
        '''
        spk_emd = os.path.join(self.database,spk_id + '.npy')
        if os.path.exists(spk_emd):
            os.remove(spk_emd)
            return True
        return False