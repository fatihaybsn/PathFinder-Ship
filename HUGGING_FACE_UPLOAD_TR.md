# Hugging Face Yayınlama Talimatı — Retraining v2

Model klasörleri güncellenen ağırlıkları içeriyor. GitHub bütün deney zincirini belgeler; Hugging Face ana kullanım modellerini ve isteğe bağlı milestone adapter'larını barındırır. Hiçbir yerel model, yayın revision'ı indirip smoke testten geçmeden silinmemelidir.

## Önerilen yayın sırası

| Öncelik | Kaynak | Hugging Face repo | Yayın kararı |
|---:|---|---|---|
| 1 | `models/flan_large_lora_second_try` içindeki LoRA dosyaları | `fatihaybsn/pathfinder-flan-t5-large-second-try-lora` | Ana final model |
| 2 | `models/minilm_intent_int8` | `fatihaybsn/pathfinder-minilm-intent-int8` | Ana intent modeli |
| 3 | Güncel Second Try ONNX export'u | `fatihaybsn/pathfinder-flan-t5-large-second-try-onnx-int8` | Yalnız yeni ağırlıklarla eşleştiği doğrulanırsa |
| 4 | `models/flan_large_lora_first_try` | `fatihaybsn/pathfinder-flan-t5-large-first-try` | İsteğe bağlı runner-up |
| 5 | `models/flan_large_lora_chat12_step1980` | `fatihaybsn/pathfinder-flan-t5-large-chat12-step1980` | İsteğe bağlı milestone |
| 6 | `models/flan_large_lora_rag2_step1320` | `fatihaybsn/pathfinder-flan-t5-large-rag2-step1320` | İsteğe bağlı task-weight deneyi |

`kötü` ve step-1485 ağırlıklarını Hugging Face'e yüklemek zorunlu değildir. GitHub sonuç tablosu ve model kartları bu denemeleri kanıt zincirinde tutar.

## Model kartları

Her repo oluşturulmadan önce ilgili `model-cards/<model>/README.md` dosyasını model klasörüne `README.md` adıyla kopyalayın. Second Try kartı şu sonuçları göstermelidir:

| Chat token-F1 | RAG token-F1 | RAG exact match |
|---:|---:|---:|
| 0.5216 | 0.8894 | 0.7938 |

Kartta ayrıca base model (`google/flan-t5-large`), LoRA parametreleri, görev ağırlıkları, test seti kapsamı ve otomatik metrik sınırlamaları bulunmalıdır.

## Komutlar

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Hugging Face web arayüzünde boş model repolarını oluşturduktan sonra:

```bash
hf upload fatihaybsn/pathfinder-flan-t5-large-second-try-lora ./models/flan_large_lora_second_try . --exclude "onnx/**"
hf upload fatihaybsn/pathfinder-minilm-intent-int8 ./models/minilm_intent_int8 .
```

ONNX dosyaları yeni Second Try ağırlıklarından export edildiyse:

```bash
hf upload fatihaybsn/pathfinder-flan-t5-large-second-try-onnx-int8 ./models/flan_large_lora_second_try/onnx .
```

## ONNX yayın kapısı

Eski parity sonucu yeni retrained model için kullanılamaz. ONNX repo'su yalnızca aşağıdakiler mevcutsa “current final” olarak yayınlanmalıdır:

1. Export'un güncel Second Try ağırlıklarından üretildiği kayıtlı.
2. Encoder, decoder ve decoder-with-past dosyalarının SHA-256 değerleri kayıtlı.
3. PyTorch+LoRA ile aynı sabit örneklerde parity sonucu mevcut.
4. Temiz ortamdan indirme ve en az üç örneklik inference testi başarılı.

## Yayın sonrası doğrulama

1. Hugging Face repo revision/commit kimliğini kaydedin.
2. `snapshot_download` ile temiz klasöre indirin.
3. Üç sabit örneklik inference smoke testi çalıştırın.
4. İndirilen dosya hash'lerini yayın öncesi hash'lerle karşılaştırın.
5. Revision URL'sini GitHub v2 sonuç kartına ekleyin.
6. Bu adımlar tamamlanmadan yerel ağırlıkları silmeyin.
