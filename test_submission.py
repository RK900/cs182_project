import json, sys
from model_sentence_lstm import ReviewPredictionModel

import torch as th
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification

from spacy.lang.en import English

with open('model-config.json', 'r') as f:
  best_accuracy_params = json.load(f)

num_sentences = 10
max_tokenized_length = 64
embedding_size = 769
lstm_hidden_size = best_accuracy_params["lstm_hidden_size"]

nlp = English()
nlp.add_pipe("sentencizer")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

embedding_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=9, state_dict=th.load(f"finetuned-bert.pt"))
embedding_bert.eval()
embedding_bert.to(device)

def get_embeds(text):
	concatted_shape = None
	pad_value = None
	
	doc = nlp(text)
	embeddeds = []
	sents = list(doc.sents)
	all_tokenized = []
	for sentence in sents[:num_sentences]:
			sentence = str(sentence)
			tokenized = tokenizer(sentence, truncation=True, padding="max_length", max_length=max_tokenized_length)[0]
			all_tokenized.append(tokenized.ids)
	
	with th.no_grad():
		sentence_tensor = th.tensor(all_tokenized).to(device)
		concatted = np.concatenate([
			# take output corresponding to CLS
			embedding_bert.bert(sentence_tensor, output_hidden_states=True, return_dict=True)[1].cpu().numpy(),
			np.zeros((len(all_tokenized), 1))
		], axis=1)
		
		if not concatted_shape:
			concatted_shape = concatted.shape
			pad_value = np.zeros(concatted_shape[1])
			pad_value[-1] = 1
		
		embeddeds += list(concatted)

	if len(sents) < num_sentences:
		embeddeds += [pad_value] * (num_sentences - len(sents))

	return embeddeds

main_model = ReviewPredictionModel(
	embedding_size, lstm_hidden_size
)
main_model.load_state_dict(th.load(f"main-model.pt"))
main_model.eval()
main_model.to(device)

def eval(text):
	embeddeds = get_embeds(text)
	in_tensor = th.tensor(embeddeds, dtype=th.float).unsqueeze(0).to(device)
	out = main_model(in_tensor)
	max_prediction = th.argmax(out, dim=1)[0].item()
	return max_prediction * 0.5 + 1

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")