# PathFinder-Ship

PathFinder-Ship; yerel öncelikli ve intent yönlendirmeli, çok modlu bir asistan mimarisidir. Sistem; instruction takip eden yanıt üretimi için int8 quantize bir ONNX üzerinde çalışan Flan-T5 Large sınıfı bir encoder–decoder modelini (projede **Passenger-Bot** olarak adlandırılmıştır) kullanır. Flan ailesi, instruction fine-tuning yaklaşımıyla geniş görev karışımlarında talimat izleme performansını artıracak şekilde tasarlanmıştır. 

Mimari; komut/senaryo yönlendirmesi için niceliksel (quantized) MiniLM tabanlı bir amaç sınıflandırıcı, skor tabanlı eşikleme ile web aramasını devreye alan hibrit bir RAG motoru (ChromaDB + SQLite FTS5/BM25) ve YOLO-NAS tabanlı bir görüntü işleme hattını birleştirir. Görüntü hattı; kamerayı açma, fotoğraf çekme, nesne algılama çalıştırma, model yorumlaması ve çıktıları e-posta ile iletme işlevlerini kapsar. Tüm bileşenler, tek bir FastAPI üzerinden; sohbet geçmişi, dosya yükleme, web araması geçişi ve isteğe bağlı ses modu içeren modern bir tek sayfalık web arayüzüyle sunulur.

Ayrıca, bu projede eğitilen Passenger-Bot (Flan-T5 Large 783M Parametre) model, instruction fine-tuning ile çok görevli (multi-task) bir kurulumda; sohbet (chat) ve RAG kullanım senaryolarını birlikte kapsayacak şekilde eğitilmiştir.

Sistemde “hazır cevap” mantığı yoktur; üretilen tüm yanıtlar, modelin talimat (instruction) takip kabiliyeti ile çalışma anında oluşturulur.

## Demo

> 🔗 **Demo video** 
🎥 [Watch on YouTube](https://youtu.be/mqfz_hPWoi0)


## İçindekiler

- [Öne Çıkan Özellikler](#öne-çıkan-özellikler)
- [Model Fine-Tuning Kanıtları](#model-fine-tuning-kanıtları)
- [Mimari Genel Bakış](#mimari-genel-bakış)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
  - [1. Klonlama ve Sanal Ortam](#1-klonlama-ve-sanal-ortam)
  - [2. Ortam Değişkenleri](#2-ortam-değişkenleri)
  - [3. RAG Corpus ve İndeks Oluşturma](#3-rag-corpus-ve-indeks-oluşturma)
  - [4. Backend ve Frontend Çalıştırma](#4-backend-ve-frontend-çalıştırma)
- [Kullanım](#kullanım)
  - [Sohbet](#sohbet)
  - [Dokümanlar Üzerinden Soru-Cevap (RAG)](#dokümanlar-üzerinden-soru-cevap-rag)
  - [Web Destekli Yanıtlar](#web-destekli-yanıtlar)
  - [Kamera, Fotoğraf ve Nesne Tespiti](#kamera-fotoğraf-ve-nesne-tespiti)
  - [Sesli Mod](#sesli-mod)
- [İç Yapı](#iç-yapı)
  - [Intent Sınıflandırıcı (MiniLM ONNX)](#intent-sınıflandırıcı-minilm-onnx)
  - [Flan-T5 ONNX Servisi](#flan-t5-onnx-servisi)
  - [Hibrit RAG Motoru](#hibrit-rag-motoru)
  - [Web Arama Entegrasyonu](#web-arama-entegrasyonu)
  - [Görüntü Pipeline ve Mail Bildirimleri](#görüntü-pipeline-ve-mail-bildirimleri)
- [Kullanım Senaryoları](#kullanım-senaryoları)
- [Sınırlamalar ve Yol Haritası](#sınırlamalar-ve-yol-haritası)
- [Katkı ve Kaynaklar](#katkı-ve-kaynaklar)
- [Ekstra Bilgiler](#ekstra-bilgiler)


## Öne Çıkan Özellikler

- **Intent routing (MiniLM, ONNX INT8)**  
  - Her kullanıcı mesajını şu niyetlerden birine sınıflandırır:  
    `open_camera`, `close_camera`, `take_photo`, `object_detect`, `chat`
  - Güveni yüksek olan komutlar doğrudan kamera / görüntü pipeline’ına yönlenir.
  - Diğer mesajlar sohbet / RAG akışına gider.

- **Kendi dokümanların üzerinde hibrit RAG**
  - SentenceTransformer (`all-MiniLM-L6-v2`) ile üretilen embedding’ler **ChromaDB** içinde tutulur.
  - Aynı chunk’lar **SQLite FTS5 (BM25)** içinde tam metin indekslenir.
  - Sorgu anında semantik ve keyword skorları normalize edilip `[0, 1]` aralığında tek bir hibrit skor halinde birleştirilir.

- **Opsiyonel web destekli cevaplar**
  - DuckDuckGo tabanlı web arama (`ddgs`).
  - HTML temizleme, chunk’lama ve basit alaka skoru hesaplama.
  - Sıkı bir “web strength” eşiği ile çalışır; yeterince güçlü değilse web bağlamı **hiç kullanılmaz**.

- **Instruction tabanlı Flan-T5 Large (ONNX INT8)**
  - Tek bir model, birden fazla davranış:
    - serbest sohbet (chat),
    - RAG bağlamına dayalı Soru-Cevap,
    - RAG zayıfken güvenli fallback cevaplar,
    - kamera ve nesne tespiti için kısa anlatımlar (narration).

- **YOLO-NAS ile görüntü araçları**
  - Tarayıcı üzerinden kamerayı aç.
  - Anlık kare al (fotoğraf çek).
  - Şu kaynaklarda nesne tespiti yap:
    - canlı kamera frame’i,
    - kullanıcının yüklediği bir görsel.
  - Bounding box ve sınıf label’larını çiz.
  - Son N görüntüyü tutan ring buffer yapısı.
  - Tespit sonuçlarını e-posta ile (ekli görsel + metin) gönder.

- **Modern tek sayfa web arayüzü**
  - Yan panelde chat geçmişi, localStorage ile kalıcı.
  - Markdown render ve syntax highlight (kod blokları).
  - Dosya yükleme ve yüklenen görsel üzerinde detection.
  - Web Search toggle (sadece lokal vs lokal+web).
  - Light/dark tema.
  - Sesli mod (tarayıcı STT + TTS, İngilizce).

Tüm çekirdek modeller ONNX INT8 formatında ve **CPU üzerinde**, ONNX Runtime ile lokal olarak çalışır.

## Model Fine-Tuning Kanıtları

Bu projedeki dil ve intent modelleri **rastgele ağırlıklardan sıfırdan eğitilmedi**. Pretrained checkpoint'ler, toplanıp düzenlenen proje verileriyle fine-tune edildi: `google/flan-t5-large`, çok görevli Chat + RAG düzeninde LoRA ile; MiniLM ise beş sınıflı intent yönlendirmesi için eğitildi. Yalnızca kazanan model değil, altı farklı Flan-T5 eğitim denemesi ve başarısız/ara sonuçlar da karşılaştırmada tutuldu.

Tüm Flan-T5 adapter'ları aynı dondurulmuş test setlerinde değerlendirildi: **300 Chat örneği** (`chat_reference_v1`) ve **160 RAG örneği** (`rag_project_v1`). `1.2x Chat` ve `2x RAG` ifadeleri **görev-loss ağırlıklarını** belirtir; model boyutunun büyütüldüğü veya verinin kopyalandığı anlamına gelmez.

| Deney | Bilinçli eğitim değişikliği | Chat token-F1 | RAG token-F1 | RAG EM | Karar |
|---|---|---:|---:|---:|---|
| LoRA q/v — `kötü` | Yalnız q/v LoRA, r=16, alpha=32 | 0.3942 | 0.8127 | 0.7063 | Reddedildi: Chat zayıf |
| RAG-loss-weight 2.0 — step 1320 | RAG ağırlığı 2.0; Chat 1.0 | 0.3778 | 0.5864 | 0.4625 | Ağırlıklandırma genellemeyi iyileştirmedi |
| Chat-loss-weight 1.2 — step 1485 | Chat ağırlığı 1.2; RAG 1.0 | 0.4695 | 0.6719 | 0.5563 | Ara checkpoint |
| Chat-loss-weight 1.2 — step 1980 | Aynı ağırlık, daha fazla step | 0.4684 | 0.6587 | 0.5438 | Step 1485'e göre küçük gerileme |
| My Class — First Try | İlk özel trainer/loss denemesi | 0.4937 | 0.8421 | 0.7438 | İkinci en iyi |
| **My Class — Second Try** | Chat ağırlığı 1.7, göreve özel smoothing, kısmi R-Drop | **0.5216** | **0.8894** | **0.7938** | **Seçilen final; üç metriğin tamamında en iyi** |

![Flan-T5 Large LoRA yeniden eğitim karşılaştırması](docs/model-evidence/retraining-v2/figures/flan_retraining_v2_dashboard.png)

Daha uzun süren 1.2x Chat eğitimi küçük bir gerileme gösterirken özel Second Try üç görev metriğinde de öne çıktı. Yapay bir birleşik skor üretilmedi. Referans örtüşmesine dayalı bu metrikler yalnızca belirtilen proje test setleri için geçerlidir ve insan değerlendirmesinin yerine geçmez.

Mevcut **MiniLM INT8 ONNX** sonucu da kanıt paketinde korundu. Model, dondurulmuş ve dengeli 1.000 örneklik proje stres setinde `1.0000` accuracy, macro-F1 ve weighted-F1 elde etti. 10-bin ECE değeri `0.1813` olduğundan, bu sette sınıf tahminleri kusursuz olsa da olasılık kalibrasyonu kusursuz değildir.

İncelenebilirlik için repoda [ayrıntılı sonuç kartı](docs/model-evidence/retraining-v2/RESULT_CARD.md), [makine tarafından okunabilir metrikler](docs/model-evidence/retraining-v2/metrics/flan_retraining_results.json), [düzenlenebilir CSV](docs/model-evidence/retraining-v2/metrics/flan_retraining_results.csv), [kanıt kaynağı açıklaması](docs/model-evidence/retraining-v2/PROVENANCE.md), görseller ve [SHA-256 artifact manifesti](docs/model-evidence/retraining-v2/artifact_manifest.json) bulunur.

> **Kapsam notu:** YOLO-NAS bu projede pretrained görüntü modeli entegrasyonu ve ONNX deployment bileşeni olarak kullanılır; repo özel YOLO-NAS eğitimi yapıldığı iddiasında bulunmaz.


## Mimari Genel Bakış

Yüksek seviyede istek akışı:

1. **Frontend**, kullanıcı mesajını `POST /api/intent` endpoint’ine gönderir.
2. **Intent sınıflandırıcı (MiniLM ONNX)** `(intent, score)` döndürür:
   - `open_camera`, `close_camera`, `take_photo`, `object_detect`, `chat`.
3. Eğer `intent` bir **komut** ise ve `score >= CLS_ROUTE_THRESHOLD`:
   - Kamera / görüntü aksiyonları tetiklenir:
     - kamera aç/kapat,
     - fotoğraf çek,
     - YOLO ile nesne tespiti,
     - görüntü kaydı ve istenirse e-posta ile gönderim.
   - Flan-T5, bu aksiyonlara dair kısa bir “onay / anlatım” cümlesi üretir.
4. Aksi halde mesaj **sohbet / QA** olarak ele alınır:
   - Kullanıcı **Web Search kapalı** ise:
     - sadece lokal RAG veya
     - RAG zayıfsa “model-only + güvenli instruction” cevabı.
   - **Web Search açık** ise:
     - web sonuçları çekilir ve skorlanır,
     - web strength’e göre web chunk’ları dahil edilir veya tamamen dışarıda bırakılır,
     - lokal-only, web-only, lokal+web veya model-only seçeneklerinden biri seçilir.
5. Flan-T5 nihai cevabı üretirken:
   - sohbet için chat_instruction,
   - RAG için sıkı rag_instruction,
   - düşük güven için fallback_instruction kullanılır.

Tek bir FastAPI backend; şu servisleri orkestre eder:

- `NLUClassifier` (MiniLM intent),
- `T5Service` (Flan-T5),
- `RAGService` (Chroma + BM25 + web),
- `YOLOService` (YOLO-NAS),
- storage / mail / prompt yardımcı katmanları.

![1758439761461](https://github.com/user-attachments/assets/d561f507-6d75-48ca-9a1f-c3032491ae3d)

![1758439761198](https://github.com/user-attachments/assets/1eb601bb-4e4a-443d-9668-44425c7f8aa2)


## Proje Yapısı

```text
PathFinder-Ship/
├─ backend/
│  ├─ main.py                 # FastAPI / uvicorn entrypoint
│  ├─ config.py               # .env → CFG dict (path, threshold, model ayarları)
│  ├─ web/
│  │  └─ app.py               # API: /api/intent, /api/chat, /api/rag, /api/photo, /api/detect, /api/upload, /api/health
│  ├─ assets/
│  │  ├─ models/
│  │  │  ├─ nlu/              # MiniLM intent ONNX + tokenizer
│  │  │  ├─ t5/               # Flan-T5 ONNX encoder/decoder + tokenizer
│  │  │  └─ yolo_nas/         # YOLO-NAS ONNX + labels.txt
│  │  └─ rag/
│  │     └─ chroma_db/        # ChromaDB persistant store + BM25 SQLite
│  ├─ data/
│  │  ├─ rag/
│  │  │  └─ corpus/           # RAG corpus (PDF/DOC/TXT)
│  │  └─ web_out/
│  │     ├─ photo/            # ring buffer photo çıktıları
│  │     └─ detect/           # ring buffer detect çıktıları
│  ├─ services/
│  │  ├─ nlu_classifier.py    # MiniLM intent classifier
│  │  ├─ t5.py                # Flan-T5 ONNX servisi
│  │  ├─ rag.py               # RAGService (hibrit arama + web kapısı + context build)
│  │  ├─ yolo.py              # YOLO-NAS ONNX wrapper (pre/post, NMS)
│  │  └─ rag_backend/
│  │     ├─ io_loader.py      # PDF/DOC/TXT → raw text
│  │     ├─ preprocess.py     # temizleme + token-bazlı chunking
│  │     ├─ indexer.py        # embedding + Chroma + FTS5 index
│  │     ├─ search.py         # hibrit retrieval (Chroma + BM25)
│  │     ├─ prompt.py         # tokenizer-bilinçli context oluşturma
│  │     └─ websearch.py      # DuckDuckGo + HTML parse + web chunk skoru
│  └─ utils/
│     ├─ text.py              # instruction ve prompt builder’lar
│     ├─ vision.py            # YOLO NMS + çizim
│     ├─ storage.py           # ring buffer kayıt yönetimi
│     └─ mailer.py            # SMTP ile görsel gönderimi
├─ frontend/
│  ├─ index.html              # SPA shell (sidebar + chat + kamera alanı)
│  ├─ styles.css              # modern responsive CSS
│  └─ app.js                  # chat/intent/camera/upload/voice mantığı
└─ requirements.txt
```

## Kurulum

### 1. Klonlama ve Sanal Ortam

```bash
git clone https://github.com/YOUR_USERNAME/pathfinder-ship.git
cd pathfinder-ship

python -m venv .venv

# Windows (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Ortam Değişkenleri

`backend/.env` dosyası oluştur ve şu şablonu temel al:

```env
# Uygulama
APP_NAME=PathFinder-Ship
DEFAULT_USER_NAME=Passenger
BOT_NAME=Passenger-Bot
DEBUG=false

# API
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_ORIGIN=http://localhost:5173

# Classifier (MiniLM-L6, ONNX INT8)
CLS_ONNX=assets/models/nlu/intent-minilm-int8.onnx
CLS_TOKENIZER_DIR=assets/models/nlu/tokenizer
CLS_MAX_LEN=64
CLS_ROUTE_THRESHOLD=0.50

# Storage (ring buffer)
PHOTO_DIR=data/web_out/photo
DETECT_DIR=data/web_out/detect
MAX_FILES_PER_DIR=10

# Email (foto/detect sonuçlarını mail atmak için)
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=1
EMAIL_USER=your_app_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
EMAIL_FROM=PathFinder-Ship <your_app_email@gmail.com>
EMAIL_TO_PHONE=destination_email@example.com

# Flan-T5 Large (ONNX INT8)
T5_TOKENIZER_DIR=assets/models/t5/tokenizer
T5_ENCODER=assets/models/t5/encoder_model_int8.onnx
T5_DECODER=assets/models/t5/decoder_model_int8.onnx
T5_MAX_SRC_LEN=512
T5_MAX_NEW_TOKENS_CHAT=256
T5_MAX_NEW_TOKENS_RAG=64

# RAG
RAG_SCORE_THRESHOLD=0.40
RAG_TOP_K=4
RAG_MAX_CTX_TOKENS=512
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PATH=assets/rag/chroma_db
VECTOR_WEIGHT=0.75
BM25_WEIGHT=0.25
RAG_WEB_MIN_STRENGTH=0.75
WEB_CHUNK_SUPPORT_THRESHOLD=0.70

# YOLO-NAS
YOLO_ONNX=assets/models/yolo_nas/yolo_nas_s_coco.onnx
YOLO_LABELS=assets/models/yolo_nas/labels.txt
YOLO_SIZE=640
YOLO_CONF=0.50
YOLO_IOU=0.50
```

### 3. RAG Corpus ve İndeks Oluşturma

1. Dokümanlarını RAG corpus klasörüne koy:

   ```text
   backend/data/rag/corpus/
   ```

   Desteklenen formatlar:
   - `.pdf`
   - `.docx`
   - `.txt`

2. `backend/` içinde indeks oluştur:

   ```bash
   cd backend
   python -m services.rag_backend.indexer --src data/rag/corpus --reset
   ```

Bu komut:

- PDF/DOCX/TXT dosyalarını okur,
- metni temizler ve token-bazlı (veya gerekiyorsa kelime-bazlı) chunk’lara böler,
- chunk’ları `all-MiniLM-L6-v2` ile embed eder,
- embedding’leri ChromaDB’ye, metni de SQLite FTS5’e yazar.

### 4. Backend ve Frontend Çalıştırma

Backend:

```bash
cd backend
python main.py
# FastAPI: http://0.0.0.0:8000
```

Frontend (basit static server):

```bash
cd frontend
python -m http.server 5173
```

Tarayıcıdan:

```text
http://localhost:5173
```

adresini aç.


## Kullanım

### Sohbet

- Input alanına mesaj yazıp **Send** butonuna bas.
- Basit sohbet senaryolarında (örneğin “Who are you?”, “Tell me a story”):
  - intent `chat` olarak sınıflanır,
  - `POST /api/chat` çağrılır,
  - Flan-T5, chat instruction ile doğal İngilizce cevap üretir.

### Dokümanlar Üzerinden Soru-Cevap (RAG)

- RAG corpus’taki PDF/DOC/TXT içeriklerine dair sorular sor:
  - “Summarize the main findings of the 2014 Kosovo report.”
  - “What does the thesis say about migration patterns?”
- Backend akışı:
  - `POST /api/rag` (`use_internet=false`).
  - Hibrit arama: Chroma + BM25.
  - En iyi skor ≥ `RAG_SCORE_THRESHOLD` ise:
    - ilgili chunk’lardan context oluşturulur ve sıkı RAG instruction ile T5’e verilir.
  - Eşik altında:
    - T5, fallback_instruction ile **context’siz** çağrılır; uydurmaması için talimatlıdır.

### Web Destekli Yanıtlar

- Arayüzde **Web Search** toggle’ını aç.
- Genel bilgi / tarih / güncel konular ile ilgili sorular sor.
- Backend akışı:
  - Lokal RAG her zamanki gibi çalışır.
  - DuckDuckGo’dan web sonuçları çekilir, temizlenir, chunk’lanır.
  - `web_strength` skoru, web parçalarının kullanılmaya değer olup olmadığını belirler.
  - Karar:
    - güçlü lokal + güçlü web → **hybrid local+web**,
    - güçlü lokal, zayıf web → **local-only**,
    - zayıf lokal, güçlü web → **web-only**,
    - ikisi de zayıf → **model-only** (fallback).

Backend ayrıca `sources` listesi döndürür; UI altında “Resources” başlığı ile gösterilebilir.

### Kamera, Fotoğraf ve Nesne Tespiti

Tipik flow’lar:

1. **Kamerayı aç**

   - “Open the camera” / “Turn on camera”.
   - Intent: `open_camera` + yüksek skor.
   - Frontend: `getUserMedia` ile video stream açılır.
   - Backend: T5 kısa bir narration üretir (“Opening the camera for you now…”).
   - Chat: onay mesajı + canlı video preview görünür.

2. **Fotoğraf çek**

   - “Take a photo”.
   - Tarayıcı mevcut frame’i alır, `POST /api/photo` endpoint’ine gönderir.
   - Backend:
     - fotoğrafı ring buffer mantığıyla `PHOTO_DIR` içine kaydeder,
     - background task ile mail gönderimi tetikler (`EMAIL_TO_PHONE`).
   - Yanıt: kaydedilen yol ve statik URL döner.

![detect_10](https://github.com/user-attachments/assets/56f46598-fc34-4b9d-a6d7-ec025bb7fdb8)


3. **Nesne tespiti**

   İki yöntem:

   - **Kameradan:**
     - Önce kamerayı aç,
     - “Object detect” de,
     - Son frame `POST /api/detect` ile gönderilir.
   - **Yüklenen görselden:**
     - Dosya upload butonundan bir görsel seç,
     - Mesaj gönder; frontend `POST /api/detect` çağırır.

   Backend:

   - YOLO-NAS ONNX modeli ile detection çalışır.
   - Detections → özet string (ör. `"2 person, 1 cell phone"`).
   - T5, bu özete göre kısa bir sahne anlatımı üretir (muhtemel senaryo; kesin iddia yok).
   - Çizili görsel `DETECT_DIR`’e ring buffer ile kaydedilir.
   - Görsel + özet + narration, background task ile mail olarak gönderilir.
   - API yanıtı:
     - `labels`, `summary`, `narration`, `image_url` gibi alanlar içerir.

### Sesli Mod

- Tarayıcı Web Speech API kullanır:
  - STT (speech-to-text) → konuşmayı yazıya çevirir,
  - TTS (text-to-speech) → cevabı sesli okur.
- **Voice** toggle’ı açıkken:
  - Text input devre dışı kalır.
  - Her transcript, `POST /api/rag` endpoint’ine gönderilir:
    - `use_internet` ve `web_only` parametreleri Web Search toggle’ına göre ayarlanır.
  - Yanıt, sesli okunur ve istenirse chat’e de yazdırılır.

> Not: Sesli mod şu an sadece Q&A (RAG + web) odaklıdır.  
> Voice input, henüz intent routing üzerinden kamera komutlarını tetiklemiyor.


## İç Yapı

### Intent Sınıflandırıcı (MiniLM ONNX)

- INT8 quantize MiniLM-L6 modeli ONNX formatında CPU üzerinde çalışır.
- Yerel HF tokenizer + config kullanır.
- Çıktı:
  - kanonik label (`open_camera`, `close_camera`, `take_photo`, `object_detect`, `chat`),
  - softmax skoru (olasılık).
- Global eşik: `CLS_ROUTE_THRESHOLD`; komut/soru ayrımını buradan yönetirsin.

### Flan-T5 ONNX Servisi

- Encoder ve decoder ayrı ONNX modelleri.
- Tokenizer yerelde (`assets/models/t5/tokenizer`).
- İki temel üretim modu:
  - **chat**: top-p sampling ile doğal sohbet,
  - **rag**: greedy, kısa cevaplar (daha fokus).
- Birden fazla prompt şablonu:
  - `chat_instruction`: bot adı, app adı ve davranış kurallarını tanımlar.
  - `rag_instruction`: cevabı sadece context içi bilgiyle sınırlar.
  - `fallback_instruction`: context yokken uydurmayı engeller.
  - Kamera/tespit için özel instruction’lar (open/close/take_photo/detection narration).

### Hibrit RAG Motoru

- `preprocess`:
  - metni temizler (soft hyphen vs.),
  - tokenizer varsa token-bazlı, yoksa kelime-bazlı chunk’lar üretir.
- `indexer`:
  - chunk’ları `all-MiniLM-L6-v2` ile embed eder,
  - embedding’leri Chroma koleksiyonuna,
  - metni ve metadata’yı SQLite FTS5 tablosuna yazar.
- `search`:
  - `chroma_search`: uzaklıklardan similarity (`1 - d`) hesaplar ve normalize eder,
  - `bm25_search`: FTS5 BM25 skorlarını alır, ters çevirir ve normalize eder,
  - `hybrid_search`: iki skoru `VECTOR_WEIGHT` ve `BM25_WEIGHT` ile birleştirir, 0–1 aralığına kırpar.
- `RAGService`:
  - `hybrid_search` → context list + best_score + kaynak listesi döndürür,
  - local vs web vs hybrid vs model-only kararını verir,
  - tokenizer-bilinçli context kesimiyle T5’e hazır hale getirir.

<img width="1919" height="1079" alt="Ekran görüntüsü 2025-10-01 201021" src="https://github.com/user-attachments/assets/3598dfdb-fe97-416f-adb6-fbe6d9dd30ff" />


### Web Arama Entegrasyonu

- DuckDuckGo üzerinden (`ddgs`) sonuç çeker.
- requests + BeautifulSoup ile sayfa içeriğini indirir ve HTML temizler.
- Hem domain hem sayfa seviyesinde noise azaltma yapılır.
- `chunk_text` + `clean_text` ile içerik küçük parçalara bölünür.
- `_norm_relevance` ile her chunk için 0–1 arası alaka skoru hesaplanır.
- `web_strength` fonksiyonu:
  - en iyi skor,
  - güçlü chunk sayısı
  üzerinden tek bir `[0, 1]` metrik üretir.
- `web_strength >= RAG_WEB_MIN_STRENGTH` değilse web bağlamı **hiç kullanılmaz**.

### Görüntü Pipeline ve Mail Bildirimleri

- YOLO-NAS ONNX → ONNX Runtime CPU.
- Preprocess:
  - letterbox resize,
  - BGR → RGB,
  - `[0, 1]` normalize,
  - NCHW, batch dimension.
- Postprocess:
  - XYXY kutular + sınıf skorları,
  - sınıf ve confidence seçimi,
  - orijinal görüntü koordinatlarına ölçekleme,
  - clipping,
  - NMS.
- Çizim:
  - OpenCV ile rect + label,
  - sabit renk ve font.
- Kayıt:
  - `save_with_ring_buffer` ile `.ring.idx` üzerinden dönen slot mantığı,
  - `photo_01.jpg`, `detect_01.jpg` … şeklinde en fazla `MAX_FILES_PER_DIR` dosya.
- Mail:
  - `send_image_via_email` tek görseli SMTP ile yollar,
  - config eksikse sessizce atlar (sistemi bozmaz).


## Kullanım Senaryoları

### 1. Kişisel Bilgi Bankası Asistanı

- Çalıştığın konuya ait PDF/DOC/TXT dokümanlarını RAG corpus’a bırak.
- Bu dokümanlar hakkında doğal dilde sorular sor.
- Sistem:
  - ilgili chunk’ları hibrit skor ile seçer,
  - mümkünse sadece bu chunk’lardan cevap çıkarır,
  - mümkün değilse “bilmiyorum / dokümanlarda yok” diyerek net olur.

### 2. Görsel Olay Kaydedici / Masaüstü Asistan

- Masaüstünde bir webcam bağlı olsun.
- Komutlar:
  - “Open camera”
  - “Take a photo”
  - “Object detect”
- Sistem:
  - kamerayı açar,
  - istenildiğinde fotoğraf çeker,
  - YOLO-NAS ile nesne tespiti yapar,
  - çizili görseli ring buffer ile saklar,
  - tespit sonuçlarını (görsel + metin) mail ile telefonuna gönderir.

### 3. Edge / Embedded AI Konsolu

- Tüm modeller CPU üzerinde ONNX Runtime ile çalıştığı için:
  - backend’i Jetson / industrial PC gibi sistemlere taşımak,
  - farklı kamera kaynaklarına uyarlamak,
  - kurumsal / domain’e özel RAG corpus’larıyla kullanmak mümkün.
- Servis katmanı (`services/*.py`) modüler:
  - model path’lerini değiştirmek,
  - threshold’ları ayarlamak,
  - yeni tool’lar (ör. ASR, OCR) eklemek kolay.


## Sınırlamalar ve Yol Haritası

- **Görsel anlatımlarda halüsinasyon riski**
  - Caption, YOLO label’larına dayanıyor; piksel düzeyinde akıl yürütme yok.
  - Bazen renk / kıyafet gibi detayları uydurabilir.
  - Güvenlik kritik senaryolarda raw detection özetini kullanmak daha doğru.

- **Zayıf CPU’larda gecikme**
  - Flan-T5 Large INT8 olsa bile ağır bir model.
  - Uzun sohbet cevaplarında düşük donanımda gecikme hissedilebilir.
  - Gelecek geliştirme: daha küçük encoder-decoder, distillation, vb.

- **Sesli mod sadece RAG akışını kullanıyor**
  - Voice input şu an doğrudan `/api/rag`’e gidiyor.
  - Kamera komutları için intent routing’e bağlanmadı.
  - Gelecek geliştirme: transcript’leri de `/api/intent` üzerinden aynı pipeline’a sokmak.

- **Tek kullanıcı, lokal prototip**
  - Auth, multi-tenant, rate limiting yok.
  - Production ortam için:
    - kullanıcı hesapları,
    - kullanıcı bazlı corpus,
    - logging / monitoring / observability katmanları eklemek gerekir.


## Katkı ve Kaynaklar

- **Flan-T5 Large** – Google / HuggingFace (ONNX INT8’e dönüştürülmüş).  
- **MiniLM-L6** – SentenceTransformers / HuggingFace (intent + embedding).  
- **ChromaDB** – vector store.  
- **SQLite FTS5** – BM25 keyword arama.  
- **YOLO-NAS** – Deci / SuperGradients (ONNX export).  
- **DuckDuckGo (ddgs)** – web arama.  
- **FastAPI** – backend framework.  
- **Vanilla HTML/CSS/JS** – frontend (highlight.js, marked.js ile).

PathFinder-Ship; intent routing, hibrit RAG ve nesne tespiti pipeline’larını tek bir sistem tasarımı içinde birleştirir.  
Amaç, hem şirketlerin görebileceği “gerçek bir sistem mimarisi” sunmak, hem de tamamen lokal çalışan pratik bir asistan sağlamaktır.

---

## Ekstra Bilgiler

* **Geliştirici**: [Fatih AYIBASAN] (Bilgisayar Mühendisliği Öğrencisi)
* **E-posta**: [fathaybasn@gmail.com]

---
