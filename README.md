# [Face-TTS Indo]

Repository ini adalah implementasi training / inferensi Face-TTS Indo.

---
## Installation

1. Install python packages
```
pip install -r requirements.txt
```

2. Build monotonic align module
```
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

---
## Preparation
1. Download trained model weights <a href="https://www.kaggle.com/datasets/riozulfandy04/facetts">disini</a>

2. Buat folder lrs3 pada data `'data/lrs3/'`

2. Download <a href="https://www.kaggle.com/datasets/riozulfandy/pblfinaldataset">dataset</a> dan simpan pada `'data/lrs3/'`

3. Extract dan simpan audio sebagai '*.wav' files di `'data/lrs3/wav'`
   ```
   python data/extract_audio.py --list_path data/lrs3/pretrain.list
   ```

---
## Inferensi

1. Siapkan gambar pada `'test/'`. Gambar adalah gambar wajah berukuran 224 x 224 piksel.
2. Ubah `'config.py'` pada bagian berikut sesuai dengan file gambar yang telah dimasukkan pada `'test/'`.
```
test_faceimg = os.getenv("test_faceimg", "test/face1.png")
```
3. Persiapkan teks untuk di sintesis
```
echo "Ini contoh nya" > test/text.txt
```
4. Jalankan inferensi.
```
python inference.py
```
5. Hasil tersimpan di `'test/'`.

--- 
## Training

1. Cek dan ubah config.py sesuai keinginan

2. Jalankan
```
python run.py
```

---
## Reference
Referensi kode dari
<a href="https://github.com/naver-ai/facetts">Face-TTS</a>,
<a href="https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS">Grad-TTS</a>, 
<a href="https://github.com/bshall/hifigan">HiFi-GAN-16k</a>, 
<a href="https://github.com/joonson/syncnet_trainer">SyncNet</a>.

Terima kasih!
