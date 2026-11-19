# Training data for the AI-detection models

Put labelled examples here (any file names), then run:

```
pip install -r requirements-train.txt
python scripts/train_ai_models.py
```

```
training_data/images/ai/     training_data/images/real/
training_data/videos/ai/     training_data/videos/real/
training_data/audio/ai/      training_data/audio/real/
```

The script prints the accuracy on files the model never saw (leave-one-file-out) and saves the models to
`app/models/ai_image.json`, `ai_video.json` and `ai_audio.json`. Restart the app to use them.
More examples = better accuracy: aim for 15+ AI and 15+ real files per type, with people, places, day/night,
originals and WhatsApp copies on both sides.

The shipped models were trained on 8 AI + 9 real images, 4 AI + 3 real videos and 2 AI + 2 real audio clips
(personal files are not included in this project).
