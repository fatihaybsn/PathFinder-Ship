# Training Data Leakage Audit

## `minilm`

- train: 4,800 rows; 26 exact-pair duplicate rows; 26 input duplicate rows.
- validation: 600 rows; 2 exact-pair duplicate rows; 2 input duplicate rows.
- test: 600 rows; 1 exact-pair duplicate rows; 1 input duplicate rows.
- train__validation: 9 exact pair overlaps; 9 input overlaps.
- train__test: 8 exact pair overlaps; 8 input overlaps.
- validation__test: 4 exact pair overlaps; 4 input overlaps.

## `flan_chat12`

- corpus: 100,000 rows; 71 exact-pair duplicate rows; 129 input duplicate rows.
- validation: 5,000 rows; 1 exact-pair duplicate rows; 1 input duplicate rows.
- train_reconstructed: 95,000 rows; 67 exact-pair duplicate rows; 119 input duplicate rows.
- corpus__validation: 4,999 exact pair overlaps; 4,999 input overlaps.
- corpus__train_reconstructed: 94,933 exact pair overlaps; 94,881 input overlaps.
- validation__train_reconstructed: 3 exact pair overlaps; 9 input overlaps.

## `flan_rag2`

- corpus: 100,000 rows; 466 exact-pair duplicate rows; 842 input duplicate rows.
- validation: 5,000 rows; 1 exact-pair duplicate rows; 1 input duplicate rows.
- train_reconstructed: 95,000 rows; 426 exact-pair duplicate rows; 777 input duplicate rows.
- corpus__validation: 4,999 exact pair overlaps; 4,999 input overlaps.
- corpus__train_reconstructed: 94,574 exact pair overlaps; 94,223 input overlaps.
- validation__train_reconstructed: 39 exact pair overlaps; 64 input overlaps.

## `flan_second_try`

- corpus: 100,000 rows; 73 exact-pair duplicate rows; 371 input duplicate rows.
- validation: 5,000 rows; 0 exact-pair duplicate rows; 0 input duplicate rows.
- train_reconstructed: 95,000 rows; 69 exact-pair duplicate rows; 348 input duplicate rows.
- corpus__validation: 5,000 exact pair overlaps; 5,000 input overlaps.
- corpus__train_reconstructed: 94,931 exact pair overlaps; 94,652 input overlaps.
- validation__train_reconstructed: 4 exact pair overlaps; 23 input overlaps.
