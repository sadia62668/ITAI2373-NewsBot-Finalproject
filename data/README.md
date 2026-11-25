# Data

| Folder | Contents |
|---|---|
| `raw/` | `BBC News Train.csv` – Kaggle competition [learn-ai-bbc](https://www.kaggle.com/c/learn-ai-bbc/data), 1,490 articles (ArticleId, Text, Category). Public dataset, ~3.3 MB. |
| `processed/` | Generated at runtime (e.g. `newsbot_dataset.csv`, the renamed/cleaned version the notebook writes). Empty in the repo. |
| `models/` | Trained models saved with `NewsClassifier.save()` (git-ignored `*.joblib`). Empty in the repo. |
| `results/` | Analysis outputs such as `newsbot_report.md` from `bot.export_report()`. Empty in the repo. |

After `data_validator.validate()` removes 50 exact duplicates, 1,440 articles remain.
