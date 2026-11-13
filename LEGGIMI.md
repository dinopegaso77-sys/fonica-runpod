# Fonica Infinity su RunPod

Il "worker" separa i brani con Demucs e trascrive il testo con Whisper, su una scheda video NVIDIA di RunPod.
Ogni utente usa il **proprio** account e il **proprio** credito RunPod.

---

## Per chi pubblica l'app (una volta sola)

1. Su github.com crea un repository chiamato **fonica-runpod** (pubblico o privato).
2. Carica tutti i file di questa cartella, compresa la cartella nascosta `.github`
   (il modo più semplice: "Add file → Upload files" e trascini dentro l'intera cartella).
3. GitHub costruisce l'immagine da solo: lo vedi nella scheda **Actions** (circa 20–30 minuti la prima volta).
4. Quando ha finito, vai sul tuo profilo GitHub → **Packages** → `fonica-runpod` → **Package settings**
   → **Change visibility** → **Public**. Serve perché RunPod degli altri utenti possa scaricarla.
5. Il nome dell'immagine è `ghcr.io/dinopegaso77-sys/fonica-runpod:latest`.
   È già scritto dentro FonicaInfinity.html: gli utenti non devono saperlo.

Ogni volta che aggiorni `handler.py` o `Dockerfile` su GitHub, l'immagine si ricostruisce da sola.
Gli utenti la ricevono automaticamente alla successiva accensione della scheda video.

---

## Per chi usa l'app (lo trovi anche dentro Fonica Infinity)

1. Crea un account su runpod.io (va bene anche l'accesso con Google).
2. Ricarica un po' di credito (con 10 $ si separano molti brani).
3. In RunPod: **Settings → API Keys → Create API Key**, permesso **Read/Write**, copia la chiave.
4. In Fonica Infinity: **Libreria → Separa un brano → RunPod**, incolla la chiave e premi **Collega**.
   L'app crea da sola il servizio sul tuo account.

La prima volta RunPod scarica il programma (10–20 minuti): il primo brano resta in coda finché è pronto.
La chiave resta solo nel tuo browser. Non condividerla con nessuno.

## Note tecniche
- Limite: MP3 fino a circa 14 MB; solo trascrizione fino a circa 7 minuti di voce.
- Il servizio si spegne da solo dopo 20 secondi di inattività: si paga solo quando lavora.
- Modelli: `htdemucs` (4 tracce), `htdemucs_6s` (anche chitarra e piano), `htdemucs_ft` (qualità più alta).
- Whisper: modello `turbo`, modificabile con la variabile d'ambiente `WHISPER_MODEL`.
- Duetti: il riconoscimento delle voci usa SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`). Le soglie si possono cambiare con le variabili `DUET_SIM` (0.72: più bassa = meno brani riconosciuti come duetto) e `BOTH_GAP` (0.06: più alta = più righe «insieme»).
- Tonalità: la modalità `shift` sposta di semitoni le tracce già separate con il filtro `rubberband` di ffmpeg (la voce conserva il suo timbro). L'app chiede da −3 a +3 toni (da −6 a +6 semitoni) e salva le tracce pronte con il brano: il browser non deve più calcolare niente quando si cambia tonalità. Lavora sul processore, non serve la scheda video.
- Video finale: la modalità `video` rifà il video registrato (sfondo, trucco, duetto, testo karaoke, logo) più veloce del tempo reale, usando tutti i processori della macchina in parallelo. Con `lipsync` controlla secondo per secondo che bocca e voce coincidano e corregge lo spostamento (fino a ±0,3 s). I file non passano dentro la richiesta: l'app li mette nel cloud R2 dell'utente e manda al worker solo indirizzi firmati che scadono dopo un'ora, quindi serve R2 collegato. Variabile `PV_PROCS` per limitare i processi.
- Suono del brano: la modalità `tone` analizza la traccia separata di chitarra o basso (servono le 6 tracce per la chitarra) e restituisce i suoni del brano: quanto è distorto, le parti con suoni diversi, delay (tempo e ripetizioni), riverbero (durata e quantità), compressione e la curva delle frequenze per l'«EQ di somiglianza». Solo calcolo sul processore con numpy (file `tone.py`), circa 10-30 secondi a brano. Chorus, phaser e flanger non sono ancora riconosciuti. Anche qui la traccia passa dal cloud R2 con un indirizzo firmato.
- Separazione «6 tracce affinate» (`model: "mix6"`): fa la separazione a 6 tracce e poi ripassa voce, basso e batteria con `htdemucs_ft`, più preciso (per esempio quando nel basso resta la cassa della batteria). Ci mette circa il doppio. Nel suono del basso la batteria rimasta viene riconosciuta e segnalata (`warn: "bleed"`).
