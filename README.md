# Speaker Recognition

![alt text](https://github.com/mailong25/spk_reg/blob/main/veri.png?raw=true)

### Requirements
```
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```

### Initialize model
```
from verify import SpeakerVerifier

verifier = SpeakerVerifier('models/SpeakerNetTune.nemo','database')
```

### Enroll new speakers
```
verifier.enrol(['test/1001_0.wav','test/1001_1.wav','test/1001_2.wav'],'1001')
```

### Verify new audios
```
verifier.verify('test/1001_3.wav','1001') -> True
verifier.verify('test/1011_0.wav','1001') -> False
```

### Note on audio format
- Wav PCM 16 bit
- 16000 sampling rate
- Duration: 4s to 8s

### Performance
Error rate: 3%

Threshold = 0.7
 + False Positive Rate: 3%
 + True Positive Rate: 97%


Threshold = 0.74
 + False Positive Rate: 1%
 + True Positive Rate: 90%


Threshold = 0.68
 + False Positive Rate: 5%
 + True Positive Rate: 98%
