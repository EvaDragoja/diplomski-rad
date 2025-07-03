# Deferred Demosaicking ROI Video Coding Framework

Ovaj projekt implementira video coding framework s deferred demosaicking pristupom. Glavna ideja je pomaknuti demosaicking s encoder strane na decoder stranu, čime se smanjuje kompleksnost kodiranja i omogućuje efikasnije kodiranje sirovih Bayer podataka.

## Sadržaj

- [Opis](#opis)
- [Značajke](#značajke)
- [Struktura projekta](#struktura-projekta)
- [Instalacija](#instalacija)
- [Korištenje](#korištenje)
- [Primjeri](#primjeri)
- [Tehnički detalji](#tehnički-detalji)
- [Autori](#autori)
- [Licenca](#licenca)

---

## Opis

Deferred demosaicking framework omogućuje:
- Kodiranje i dekodiranje videa iz sirovih Bayer .DNG slika
- Pomak demosaickinga na dekoder (što ubrzava kodiranje i omogućuje fleksibilniju obradu)
- Usporedbu standardnog i artificial pristupa

## Značajke

- Podrška za .DNG ulazne slike (RAW Bayer)
- Automatska konverzija u YUV i MP4
- Demosaicking, white balance, gamma korekcija i upsampling na dekoderu
- Logovi i metrika kodiranja

## Struktura projekta

```
.
├── artificial_encoder.py      # Kodiranje umjetnim pristupom
├── artificial_decoder.py      # Dekodiranje umjetnim pristupom
├── normal_coding.py           # Standardni workflow
├── VIDEOS/DNG/                # Ulazni folder .DNG slike
├── *.yuv, *.mp4               # Izlazne i privremene video datoteke
└── README.md
```

## Instalacija

1. **Kloniraj repozitorij:**
   ```bash
   git clone <repo-url>
   cd path_do_python_skripti
   ```

2. **Instaliraj ovisnosti:**
   - Python 3.8+
   - [FFmpeg](https://ffmpeg.org/) (mora biti u PATH-u)
   - Ostale Python biblioteke:
     ```bash
     pip install numpy opencv-python rawpy
     ```


## Korištenje

### 1. Priprema ulaznih podataka

Stavit .DNG slike u `VIDEOS/DNG/` ili poddirektorij.

### 2. Pokretanje kodiranja i dekodiranja


#### Ručno (primjer za artificial workflow):

```bash
python artificial_encoder.py <path_to_input_folder> <output_file.mp4> -r <width>x<height>
python artificial_decoder.py <output_file.mp4> <output_file.yuv> -r <width>x<height>
```

### 3. Rezultati

- Izlazni video: `final_artificial_output.mp4`
- Izlazni YUV: `final_artificial_output.yuv`
- Logovi i metrike: `artificial_encoding_metrics.txt`, `normal_encoding_metrics.txt`

## Primjeri

```bash
python artificial_encoder.py VIDEOS/DNG/D008_C006_20210714_R1 artificial_encoded_video.mp4 -r 3840x2160
python artificial_decoder.py artificial_encoded_video.mp4 final_artificial_output.yuv -r 3840x2160
```

## Tehnički detalji

- **Artificial coding:** Demosaicking, white balance, gamma i upsampling se rade na dekoderu, čime se rasterećuje encoder i omogućuje brže kodiranje.
- **Format:** Ulazne slike su .DNG (RAW Bayer), a izlaz je YUV/MP4.
- **Upotreba FFmpeg-a:** Za konverziju između formata i kodiranje/dekodiranje.

## Autori

- Eva Dragoja

## Licenca

Ovaj projekt rađen je za potrebe diplomskog rada.
