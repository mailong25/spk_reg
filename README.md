# spk_reg
Speaker Recognition


![alt text](https://github.com/mailong25/spk_reg/blob/main/veri.png?raw=true)

```
from verify import SpeakerVerifier

verifier = SpeakerVerifier('models/SpeakerNetTune.nemo','database')

verifier.enrol(['test/1001_0.wav','test/1001_1.wav','test/1001_2.wav'],'1001')

verifier.verify('test/1001_3.wav','1001') -> True

verifier.verify('test/1011_0.wav','1001') -> False
