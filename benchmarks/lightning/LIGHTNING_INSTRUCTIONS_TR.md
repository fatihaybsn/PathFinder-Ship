# Lightning AI Çalıştırma Talimatı

## Studio gereksinimi

- Önerilen GPU: NVIDIA L4 veya A10G, 24 GB VRAM. Daha güçlü GPU kullanılabilir.
- Asgari alternatif: T4, 16 GB VRAM; özellikle IFEval ve beam-4 daha uzun sürer.
- RAM: en az 16 GB.
- Boş disk: en az 40 GB.
- İnternet: Hugging Face base modelleri, IFEval kodu, RAGBench ve COCO için açık olmalı.

## Yükleme

1. Masaüstündeki `lightning_upload/pathfinder_benchmark_bundle` klasörünün tamamını Lightning Studio'ya yükleyin.
2. Klasör yapısını değiştirmeyin; `models`, `benchmarks` ve `UPLOAD_MANIFEST.json` aynı kökte kalmalıdır.
3. Kök dizindeki `PathFinder_Lightning_Benchmark.ipynb` dosyasını açın.
4. Hücreleri yukarıdan aşağıya çalıştırın.

Notebook tamamlanan deneyleri durum dosyalarından tanır. Studio kapanırsa aynı notebook ve aynı kalıcı disk üzerinde kaldığınız yerden devam edebilirsiniz. Belirli bir deneyi bilinçli olarak tekrar çalıştırmak için `run_experiment("deney_kimliği", force=True)` kullanın.

## Başarılı tamamlanma

Son hücre aşağıdakileri yazdırmalıdır:

```text
TAMAMLANDI
İndirilecek ZIP: .../pathfinder_results_<run_id>.zip
```

Bir model hata alsa bile ZIP oluşturulur. Hata `status/<deney>.json` ve `logs/benchmark.log` içinde tutulur.

## Windows'a dönüş

Oluşturulan ZIP'i indirin ve değiştirmeden şu klasöre koyun:

```text
%USERPROFILE%\Desktop\PathFinder-Model-Evidence\lightning_results_incoming
```

ZIP içindeki TXT, JSON, CSV ve PNG dosyalarını tek tek elle düzenlemeyin. Manifest doğrulaması için orijinal ZIP gereklidir.
