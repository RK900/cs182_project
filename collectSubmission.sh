rm CookieMonster.zip
rm -rf CookieMonster
mkdir CookieMonster
papermill ablation_study.ipynb ablation_study.ipynb --log-output
cp requirements.txt CookieMonster/
cp test_submission.py CookieMonster/
cp model_sentence_lstm.py CookieMonster/
cp finetuned-bert.pt CookieMonster/
cp main-model.pt CookieMonster/
cp model-config.json CookieMonster/
zip -r CookieMonster.zip CookieMonster
