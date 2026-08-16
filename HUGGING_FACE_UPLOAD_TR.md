# Hugging Face Yayınlama Talimatı

Bu adımlar Benchmark v1 sonuçları doğrulandıktan sonra uygulanmalıdır. Henüz hiçbir modeli silmeyin.

## Önerilen depolar

| Kaynak klasör | Hugging Face repo adı |
|---|---|
| `models/flan_large_lora_second_try` içindeki LoRA dosyaları | `fatihaybsn/pathfinder-flan-t5-large-second-try-lora` |
| `models/flan_large_lora_second_try/onnx` | `fatihaybsn/pathfinder-flan-t5-large-second-try-onnx-int8` |
| `models/minilm_intent_int8` | `fatihaybsn/pathfinder-minilm-intent-int8` |
| `models/flan_large_lora_chat12_step1980` | `fatihaybsn/pathfinder-flan-t5-large-chat12-step1980` |
| `models/flan_large_lora_rag2_step1320` | `fatihaybsn/pathfinder-flan-t5-large-rag2-step1320` |

## Komutlar

Önce Hugging Face CLI kurulumu ve giriş:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Her repo için web arayüzünde boş bir model reposu oluşturun. Ardından örnekteki klasör/repo adını ilgili satıra göre değiştirin:

```bash
hf upload fatihaybsn/pathfinder-flan-t5-large-second-try-lora ./models/flan_large_lora_second_try . --exclude "onnx/**"
hf upload fatihaybsn/pathfinder-flan-t5-large-second-try-onnx-int8 ./models/flan_large_lora_second_try/onnx .
hf upload fatihaybsn/pathfinder-minilm-intent-int8 ./models/minilm_intent_int8 .
hf upload fatihaybsn/pathfinder-flan-t5-large-chat12-step1980 ./models/flan_large_lora_chat12_step1980 .
hf upload fatihaybsn/pathfinder-flan-t5-large-rag2-step1320 ./models/flan_large_lora_rag2_step1320 .
```

Second Try LoRA deposuna `onnx/` alt klasörünü yüklemeyin; ONNX için ayrı repo kullanın. Hazır model kartlarını ilgili repoya `README.md` olarak koyun.

## Yayın sonrası doğrulama

1. Repo sayfasındaki revision/commit kimliğini kaydedin.
2. Dosya SHA-256 değerlerini `UPLOAD_MANIFEST.json` ile karşılaştırın.
3. Temiz bir Lightning Studio açın.
4. `snapshot_download` ile modeli tekrar indirin.
5. Üç sabit örneklik inference smoke testi çalıştırın.
6. İndirilen revision ve smoke-test sonucu GitHub artifact tablosuna eklenmeden yerel ağırlıkları silmeyin.
