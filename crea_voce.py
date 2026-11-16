# Crea i file audio della voce del tutorial (voce maschile italiana "Nicola" del modello libero Kokoro, licenza Apache 2.0).
# Lo esegue GitHub da solo quando cambia voce/testi.json: crea solo i file che mancano e toglie quelli non più usati.
import json, os, subprocess, numpy as np, soundfile as sf
from kokoro import KPipeline
D = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(D, 'testi.json'), encoding='utf-8'))
voluti = {e['h']: e['t'] for e in L}
for f in os.listdir(D):
    if f.endswith('.mp3') and f[:-4] not in voluti:
        os.remove(os.path.join(D, f))
mancano = [(h, t) for h, t in voluti.items() if not os.path.exists(os.path.join(D, h + '.mp3'))]
print('Da creare:', len(mancano))
if mancano:
    pipe = KPipeline(lang_code='i', repo_id='hexgrad/Kokoro-82M')
    for n, (h, t) in enumerate(mancano, 1):
        parti = [a.numpy() if hasattr(a, 'numpy') else np.asarray(a) for _, _, a in pipe(t, voice='im_nicola', speed=1.0)]
        wav = os.path.join(D, h + '.wav')
        sf.write(wav, np.concatenate(parti), 24000)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', wav, '-ac', '1', '-b:a', '64k', os.path.join(D, h + '.mp3')], check=True)
        os.remove(wav)
        print(n, '/', len(mancano), h)
