# Lightning AI — Focused Evidence v1 Çalıştırma Talimatı

## Studio gereksinimi

- Önerilen GPU: NVIDIA H100 80 GB (ekrandaki makine) veya L4/A10G 24 GB.
- H100 tahmini: yaklaşık 20–45 dakika; model indirme ve ilk kurulum hızına göre 60 dakikaya yaklaşabilir.
- L4/A10G tahmini: yaklaşık 45–100 dakika.
- RAM: en az 16 GB.
- Boş disk: en az 15 GB.
- İnternet: `google/flan-t5-large` ve metrik bağımlılıklarını indirmek için açık olmalı.

Bu akış YOLO/COCO, IFEval, RAGBench, eski Small/Base modeller ve beam-4 testini çalıştırmaz. Yalnızca tek MiniLM modeli ile `Primee/Models` altındaki altı LoRA denemesini aynı 300 Chat ve 160 proje RAG örneğinde karşılaştırır. Final Second Try ONNX sürümü her görevden 25 örnekle ayrıca doğrulanır.

## Yükleme

1. Masaüstündeki `lightning_upload/pathfinder_benchmark_bundle` klasörünün tamamını Lightning Studio'ya yükleyin.
2. Klasör yapısını değiştirmeyin; `models`, `benchmarks` ve `UPLOAD_MANIFEST.json` aynı kökte kalmalıdır.
3. Kök dizindeki `PathFinder_Lightning_Benchmark.ipynb` dosyasını açın.
4. Hücreleri yukarıdan aşağıya çalıştırın.

Notebook tamamlanan deneyleri durum dosyalarından tanır. Studio kapanırsa aynı notebook ve aynı kalıcı disk üzerinde kaldığınız yerden devam edebilirsiniz. Bir modeli bilinçli olarak tekrar çalıştırmak için `run_experiment("deney_kimliği", force=True)` kullanın.

## Başarılı tamamlanma

Son hücre aşağıdakileri yazdırmalıdır:

```text
TAMAMLANDI
İndirilecek ZIP: .../pathfinder_focused_results_<run_id>.zip
```

Bir model hata alsa bile ZIP oluşturulur. Hata `status/<deney>.json` ve `logs/benchmark.log` içinde tutulur.

## Windows'a dönüş

Oluşturulan ZIP'i indirin ve değiştirmeden şu klasöre koyun:

```text
%USERPROFILE%\Desktop\PathFinder-Model-Evidence\lightning_results_incoming
```

ZIP içindeki TXT, JSON, CSV ve PNG dosyalarını tek tek elle düzenlemeyin. Manifest doğrulaması için orijinal ZIP gereklidir.
