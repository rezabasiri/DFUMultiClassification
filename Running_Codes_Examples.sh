python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 1 \
  --n-evaluations 30 \
  --device-mode single \
  --min-minority-retention 0.5 \
  --track-misclass valid \
  --phase2-only


  python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata depth_rgb depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 5 \
  --phase2-cv-folds 2 \
  --phase1-n-runs 5 \
  --n-evaluations 30 \
  --device-mode multi \
  --min-minority-retention 0.7 \
  --track-misclass both \
  --phase2-data-percentage 100 \
  --phase1-data-percentage 100 \

  python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata \
  --phase2-modalities "metadata" \
  --phase1-cv-folds 1 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 10 \
  --n-evaluations 10 \
  --device-mode multi \
  --min-minority-retention 0.5 \
  --track-misclass both \
  --phase2-data-percentage 100 \
  --phase1-data-percentage 100


  python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata \
  --phase2-modalities "metadata" \
  --phase1-cv-folds 1 \
  --phase1-n-runs 2 \
  --phase1-only \
  --device-mode multi \
  --track-misclass both \
  --phase1-data-percentage 100

  python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities depth_map thermal_map \
  --phase2-modalities "metadata+depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 5 \
  --phase2-cv-folds 5 \
  --phase1-n-runs 5 \
  --n-evaluations 30 \
  --device-mode multi \
  --min-minority-retention 0.7 \
  --track-misclass both \
  --phase2-data-percentage 100 \
  --phase1-data-percentage 100


  python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities depth_rgb depth_map thermal_map \
  --phase2-modalities "depth_rgb+depth_map+thermal_map" \
  --phase1-cv-folds 5 \
  --phase2-cv-folds 5 \
  --phase1-n-runs 5 \
  --n-evaluations 30 \
  --device-mode multi \
  --min-minority-retention 0.7 \
  --track-misclass both \
  --phase2-data-percentage 100 \
  --phase1-data-percentage 100


python src/main.py --mode search --data_percentage 100 \
  --cv_folds 3 --device-mode cpu \
  --track-misclass valid --resume_mode fresh \
  --verbosity 2

python src/main.py --mode search --data_percentage 100 \
  --cv_folds 3 --device-mode cpu \
  --track-misclass valid --resume_mode fresh \
  --verbosity 2

  python scripts/auto_polish_dataset_v2.py \
  --phase1-modalities metadata \
  --phase2-modalities "metadata" \
  --phase1-cv-folds 3 \
  --phase2-cv-folds 3 \
  --phase1-n-runs 10 \
  --n-evaluations 20 \
  --device-mode cpu \
  --min-minority-retention 0.7 \
  --track-misclass both \
  --phase2-data-percentage 100 \
  --phase1-data-percentage 100