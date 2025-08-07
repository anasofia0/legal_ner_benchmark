from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from time import time
import spacy
from spacy.training import Example
import pandas as pd
from transformers import pipeline, AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch
import seaborn as sns
from seqeval.metrics import classification_report
from sklearn.metrics import cohen_kappa_score


SPACY_LABEL_MAP = {
    'O': 'O',
    'PERSON': 'PERSON',
    'BUSINESS': 'ORG',
    'GOVERNMENT': 'ORG',
    'LEGISLATION/ACT': 'LAW',
    'LOCATION': 'GPE',
    'LOC': 'GPE',
    'MISCELLANEOUS': 'MISC',
    'COURT': 'ORG',
    'P': 'O'
}

TRANSFORMER_LABEL_MAP = {
    'O': 'O',
    'PERSON': 'PER',
    'BUSINESS': 'ORG',
    'GOVERNMENT': 'ORG',
    'LEGISLATION/ACT': 'LAW',
    'LOCATION': 'LOC',
    'LOC': 'LOC',
    'MISCELLANEOUS': 'MISC',
    'COURT': 'ORG',
    'P': 'O'
}


BENCHMARK_MODELS = {
    "XML-RoBERTa": "Davlan/xlm-roberta-base-ner-hrl",
    "BERT-Davlan": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "Legal-BERT": "nlpaueb/legal-bert-base-uncased",
    "spaCy": "en_core_web_sm"
}

def inference_report(true_tags_all, pred_tags_all):
    true_tags_flat = [tag for doc in true_tags_all for tag in doc]
    pred_tags_flat = [tag for doc in pred_tags_all for tag in doc]

    kappa_global = cohen_kappa_score(true_tags_flat, pred_tags_flat)

    entity_types = set(tag for tag in true_tags_flat if tag != 'O')

    kappa_per_type = {}
    for entity in entity_types:
        y_true = [1 if t == entity else 0 for t in true_tags_flat]
        y_pred = [1 if p == entity else 0 for p in pred_tags_flat]
        kappa_per_type[entity] = cohen_kappa_score(y_true, y_pred)

    print(set(true_tags_flat))
    print(set(pred_tags_flat))
    seqeval_report = classification_report(true_tags_all, pred_tags_all, output_dict=True)

    for key in seqeval_report:
        if isinstance(seqeval_report[key], dict):
            for subkey in seqeval_report[key]:
                if isinstance(seqeval_report[key][subkey], (np.float32, np.float64)):
                    seqeval_report[key][subkey] = float(seqeval_report[key][subkey])
                elif isinstance(seqeval_report[key][subkey], (np.int32, np.int64)):
                    seqeval_report[key][subkey] = int(seqeval_report[key][subkey])

    metrics = {
        'seqeval': seqeval_report,
        'kappa': {
            'global': kappa_global,
            'per_type': kappa_per_type
        }
    }

    return metrics

def plot_report(results, dataset_name="E-NER Dataset"):
    if not results:
        print("No results to plot")
        return

    plot_data = []
    for model_name, metrics in results.items():
        seqeval = metrics['seqeval']
        kappa = metrics['kappa']

        # Get entity types (excluding averages)
        entity_types = [k for k in seqeval.keys() if k not in ['micro avg', 'macro avg', 'weighted avg']]

        for entity in entity_types:
            plot_data.append({
                'Model': model_name,
                'Entity': entity,
                'Precision': seqeval[entity]['precision'],
                'Recall': seqeval[entity]['recall'],
                'F1': seqeval[entity]['f1-score'],
                'Kappa': kappa['per_type'].get(entity, 0)
            })

    df = pd.DataFrame(plot_data)

    # Create plots
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'NER Performance Comparison on {dataset_name}', y=1.02)

    # Plot 1: F1 Scores by Entity
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x='Entity', y='F1', hue='Model')
    plt.title('F1 Score by Entity Type')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Plot 2: Kappa Scores by Entity
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='Entity', y='Kappa', hue='Model')
    plt.title('Cohen\'s Kappa by Entity Type')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Plot 3: Precision-Recall Comparison
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='Precision', y='Recall', hue='Model', style='Entity', s=100)
    plt.title('Precision vs Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # Plot 4: Macro Averages Comparison
    macro_data = []
    for model_name, metrics in results.items():
        seqeval = metrics['seqeval']
        macro_data.append({
            'Model': model_name,
            'Macro F1': seqeval['macro avg']['f1-score'],
            'Macro Precision': seqeval['macro avg']['precision'],
            'Macro Recall': seqeval['macro avg']['recall'],
            'Avg Kappa': np.mean(list(metrics['kappa']['per_type'].values()))
        })

    macro_df = pd.DataFrame(macro_data).melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.subplot(2, 2, 4)
    sns.barplot(data=macro_df, x='Metric', y='Score', hue='Model')
    plt.title('Macro Average Metrics')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('./teste.png')

def save_experiment_report(model, exp_time, inference_report, path='./experiments/'):
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%d-%m-%y_%H-%M-%S')

    exp_report = {'model': model,
                  'inference_time': exp_time,
                  'inference_report': inference_report
                  }

    with open(path+model+'_'+timestamp, 'w') as f:
        json.dump(exp_report, f)

def spacy_ner(text: str, model):
    doc = model(text)
    entities = []

    entities = [{
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        } for ent in doc.ents]

    return entities

def transformers_ner(text: str, pipeline):
    raw_entities = pipeline(text)

    entities = [{
            "text": entity['word'],
            "label": entity['entity_group'],
            "start": entity['start'],
            "end": entity['end']
        } for entity in raw_entities]

    return entities

def convert_to_word_level(doc_df, pred_entities, model):
    if model == 'spaCy':
        feature_map = SPACY_LABEL_MAP
    else:
        feature_map = TRANSFORMER_LABEL_MAP

    words = doc_df.iloc[:, 0].tolist()
    true_tags = [feature_map[i] for i in doc_df.iloc[:, 1].tolist()]
    text = ' '.join(words)
    word_offsets = []
    pos = 0

    for word in words:
        word_offsets.append((pos, pos + len(word)))
        pos += len(word) + 1

    pred_tags = ['O'] * len(words)

    for entity in pred_entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = entity['label']

        # find which words are covered by this entity
        covered_words = []
        for i, (start, end) in enumerate(word_offsets):
            if not (end <= entity_start or start >= entity_end):
                covered_words.append(i)

        if covered_words:
            for i in covered_words:
                pred_tags[i] = entity_label

    return true_tags, pred_tags

def benchmark_models(docs, eval_docs):
    results = {}

    for model_name, model_path in BENCHMARK_MODELS.items():
        print(f'Benchmarking {model_name}')

        true_tags_all = []
        pred_tags_all = []

        if model_name == 'spaCy':
            model = spacy.load(model_path)

            start = time()
            for doc_df in eval_docs:
                words = doc_df.iloc[:, 0].tolist()
                text = " ".join(words)

                preds = spacy_ner(text, model)

                true_tags, pred_tags = convert_to_word_level(doc_df, preds, 'spaCy')
                true_tags_all.append(true_tags)
                pred_tags_all.append(pred_tags)
            exp_time = time() - start

        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )

            start = time()
            for doc_df in eval_docs:
                words = doc_df.iloc[:, 0].tolist()
                text = " ".join(words)

                preds = transformers_ner(text, ner_pipeline)

                true_tags, pred_tags = convert_to_word_level(doc_df, preds, model_name)
                true_tags_all.append(true_tags)
                pred_tags_all.append(pred_tags)
            exp_time = time() - start
        
        results[model_name] = inference_report(true_tags_all, pred_tags_all)
        results[model_name]['inference_time'] = exp_time
        print(results)
        print(f"Results for {model_name}:")
        print(json.dumps(results[model_name], indent=2))

    return results

def prepare_spacy_training_data(docs_df):
    training_data = []
    for doc_df in docs_df:
        words = doc_df.iloc[:, 0].tolist()
        tags = doc_df.iloc[:, 1].tolist()

        entities = []
        current_entity = None

        for i, (word, tag) in enumerate(zip(words, tags)):
            if tag != 'O':
                if current_entity and current_entity['label'] == SPACY_LABEL_MAP.get(tag, 'O'):
                    current_entity['end'] = i+1
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'start': i,
                        'end': i+1,
                        'label': SPACY_LABEL_MAP.get(tag, 'O')
                    }
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        training_data.append((' '.join(words), {'entities': entities}))

    return training_data

def train_spacy_model(docs_df, n_iter=10):
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for label in set(SPACY_LABEL_MAP.values()):
        if label != 'O':
            ner.add_label(label)

    training_data = prepare_spacy_training_data(docs_df)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for itn in range(n_iter):
            losses = {}
            random.shuffle(training_data)

            for text, annotations in tqdm(training_data):
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)

            print(f"Iteration {itn}, Losses: {losses}")

    return nlp
"""
def pepare_tranformer_training_data(docs_df):
    feature_map = TRANSFORMER_LABEL_MAP
    labels = feature_map.values()
    label2id = {label: i for i, label in enumerate(labels)}

    data = {'tokens': [], 'ner_tags': []}

    for doc_df in docs_df:
        words = doc_df.iloc[:, 0].tolist()
        tags = doc_df.iloc[:, 1].tolist()
        tag_ids = [label2id[feature_map[tag]] for tag in tags]

        data['tokens'].append(words)
        data['ner_tags'].append(tag_ids)

    return Dataset.from_dict(data)

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_legalbert_model(docs_df, n_epochs=3):
    tokenizer = BertTokenizerFast.from_pretrained('nlpaueb/legal-bert-base-uncased')
    
    label_list = set(TRANSFORMER_LABEL_MAP.values())
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    print(label2id)
    
    model = BertForTokenClassification.from_pretrained(
        'nlpaueb/legal-bert-base-uncased',
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    dataset = prepare_legalbert_training_data(docs_df)
    
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    training_args = TrainingArguments(
        output_dir='./legalbert-ner',
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return model, tokenizer
"""

def load_e_ner(path: str):
    df = pd.read_csv(path).fillna('')
    df.iloc[:,1] = df.iloc[:,1].apply(lambda x: x.split('-')[-1])
    split_idx = df.index[df.iloc[:,0] == '-DOCSTART-'].to_list()
    list_dfs = [df.iloc[split_idx[i]+1:split_idx[i+1]].reset_index(drop=True) for i in range(len(split_idx)-1)]
    return list_dfs

"""
def process_e_ner(docs, model):
    texts = [' '.join(doc.iloc[:, 0]) for doc in docs]
    entities, exp_time = entity_recognition(model, texts)
    return entities, exp_time

def eval_e_ner(model, docs_df, doc_entities):
    all_true = []
    all_pred = []

    for doc_df, preds in zip(docs_df, doc_entities):
        true_tags, pred_tags = convert_to_word_level(doc_df, preds, model)
        all_true.append(true_tags)
        all_pred.append(pred_tags)

    # filter documents with no entities
    filtered_true = []
    filtered_pred = []
    for t, p in zip(all_true, all_pred):
        if any(tag != 'O' for tag in t) or any(tag != 'O' for tag in p):
            filtered_true.append(t)
            filtered_pred.append(p)

    return inference_report(filtered_true, filtered_pred)
"""

if __name__ == '__main__':
    path = './data/E-NER-Dataset/all.csv'
    docs = load_e_ner(path)
    train_docs = docs[:46]
    eval_docs = docs[46:]

    """
    for model in ['legalBERT', 'spacy'][:1]:
        print(f"Training {model} model...")

        if model == 'spacy':
            nlp = train_spacy_model(train_docs, n_iter=3)
            nlp.to_disk(f"./{model}_trained_model")
        elif model == 'legalBERT':
            model_bert, tokenizer = train_legalbert_model(train_docs, n_epochs=1)
            model_bert.save_pretrained(f"./{model}_trained_model")
            tokenizer.save_pretrained(f"./{model}_tokenizer")

        print(f'{model}: Evaluation')
        if model == 'spacy':
            nlp = spacy.load(f"./{model}_trained_model")
        elif model == 'legalBERT':
            ner = pipeline('ner',
                         model=f"./{model}_trained_model",
                         tokenizer=f"./{model}_trained_model",
                         aggregation_strategy='simple')

        entities, exp_time = process_e_ner(eval_docs, model)
        results = eval_e_ner(model, eval_docs, entities)
        print(json.dumps(results, indent=2))
        save_experiment_report(model, exp_time, results)
    """

    benchmark_results = benchmark_models(train_docs, eval_docs)
    plot_report(benchmark_results)
