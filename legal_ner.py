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

GT_LABEL_MAP = {
    'O': 'O',
    'PERSON': 'PER',
    'BUSINESS': 'ORG',
    'GOVERNMENT': 'ORG',
    'LEGISLATION/ACT': 'LAW',
    'LOC': 'LOC',
    'LOCATION': 'LOC',
    'MISCELLANEOUS': 'MISC',
    'COURT': 'ORG',
    'P': 'O'
}

MODEL_LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "PER": "PER",
    "LAW": "LAW",
    "MISC": "MISC"
}


BENCHMARK_MODELS = {
    "XML-RoBERTa": "Davlan/xlm-roberta-base-ner-hrl",
    "BERT-Davlan": "Davlan/bert-base-multilingual-cased-ner-hrl",
    "spaCy": "en_core_web_sm"
}

def inference_report(true_tags_all, pred_tags_all):
    true_tags_flat = [tag for doc in true_tags_all for tag in doc]
    pred_tags_flat = [tag for doc in pred_tags_all for tag in doc]

    kappa_global = cohen_kappa_score(true_tags_flat, pred_tags_flat)

    entity_types = set(tag.split('-')[-1] for tag in true_tags_flat if tag != 'O')

    kappa_per_type = {}
    for entity in entity_types:
        y_true = [1 if entity in t else 0 for t in true_tags_flat]
        y_pred = [1 if entity in p else 0 for p in pred_tags_flat]
        kappa_per_type[entity] = cohen_kappa_score(y_true, y_pred)

    print(set(true_tags_flat))
    print(set(pred_tags_flat))
    seqeval_report = classification_report(true_tags_all, pred_tags_all, output_dict=True)

    # converting values to int and float to avoid error while saving json
    for key, value in seqeval_report.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (np.float32, np.float64)):
                    seqeval_report[key][subkey] = float(subvalue)
                elif isinstance(seqeval_report[key][subkey], (np.int32, np.int64)):
                    seqeval_report[key][subkey] = int(subvalue)

    metrics = {
        'seqeval': seqeval_report,
        'kappa': {
            'global': kappa_global,
            'per_type': kappa_per_type
        }
    }

    return metrics

def plot_report(results, dataset_name='E-NER Dataset'):
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
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%d-%m-%y_%H-%M-%S')
    plt.savefig('./experiments/plot_'+timestamp+'.png')

def save_experiment_report(inference_report, path='./experiments/'):
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%d-%m-%y_%H-%M-%S')

    with open(path+'inference_report_'+timestamp, 'w') as f:
        json.dump(inference_report, f)

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

def convert_to_word_level(doc_df, pred_entities):
    words = doc_df.iloc[:, 0].tolist()
    true_tags = doc_df.iloc[:, 1].tolist()

    word_offsets = []
    pos = 0
    for word in words:
        word_offsets.append((pos, pos + len(word)))
        pos += len(word) + 1 # space

    pred_tags = ['O'] * len(words)

    for entity in pred_entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = MODEL_LABEL_MAP.get(entity['label'], 'O')

        # find which words are covered by this entity
        covered_words = []
        for i, (start, end) in enumerate(word_offsets):
            if not (end <= entity_start or start >= entity_end):
                covered_words.append(i)
        
        # apply IOB2 format
        if covered_words:
            first_word = covered_words[0]
            pred_tags[first_word] = 'B-' + entity_label

            for i in covered_words[1:]:
                pred_tags[i] = 'I-' + entity_label

    return true_tags, pred_tags

def benchmark_models(eval_docs, models):
    results = {}

    for model_name, model_path in models.items():
        print(f'Benchmarking {model_name}')

        true_tags_all = []
        pred_tags_all = []

        if 'spacy' in model_name.lower():
            model = spacy.load(model_path)
            start = time()

            for doc_df in tqdm(eval_docs, desc=f'Evaluating {model_name}'):
                words = doc_df.iloc[:, 0].tolist()
                text = ' '.join(words)
                preds = spacy_ner(text, model)
                
                true_tags, pred_tags = convert_to_word_level(doc_df, preds)
                true_tags_all.append(true_tags)
                pred_tags_all.append(pred_tags)
            exp_time = time() - start

        else:
            print(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            ner_pipeline = pipeline(
                'ner',
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy='simple',
                device=0 if torch.cuda.is_available() else -1
            )

            start = time()
            for doc_df in tqdm(eval_docs, desc=f'Evaluanting {model_name}'):
                words = doc_df.iloc[:, 0].tolist()
                text = ' '.join(words)
                preds = transformers_ner(text, ner_pipeline)

                true_tags, pred_tags = convert_to_word_level(doc_df, preds)
                true_tags_all.append(true_tags)
                pred_tags_all.append(pred_tags)
            exp_time = time() - start
        
        report = inference_report(true_tags_all, pred_tags_all)
        report['inference_time'] = exp_time
        results[model_name] = report

        print(f"Results for {model_name}:")
        print(json.dumps(results[model_name], indent=2))

    return results

def prepare_spacy_training_data(docs_df):
    training_data = []
    for doc_df in docs_df:
        words = doc_df.iloc[:, 0].tolist()
        tags = doc_df.iloc[:, 1].tolist()
        text = ' '.join(words)
        entities = []
        current_label = ''
        entity_start_char = -1
        char_idx = 0

        for i, word in enumerate(words):
            tag = tags[i]
            
            if tag.startswith('B-'):
                if entity_start_char != -1:
                    entity_end_char = char_idx - 1
                    entities.append((entity_start_char, entity_end_char, current_label))
                entity_start_char = char_idx
                current_label = tag.split('-', 1)[1]

            elif tag.startswith('I-'):
                label = tag.split('-', 1)[1]
                if label != current_label or entity_start_char == -1:
                    if entity_start_char != -1:
                        entity_end_char = char_idx - 1
                        entities.append((entity_start_char, entity_end_char, current_label))
                    entity_start_char = -1
            
            else: # tag is 'O'
                if entity_start_char != -1:
                    entity_end_char = char_idx - 1
                    entities.append((entity_start_char, entity_end_char, current_label))
                entity_start_char = -1
                current_label = ''
                
            char_idx += len(word) + 1

        if entity_start_char != -1:
            entity_end_char = char_idx - 1
            entities.append((entity_start_char, entity_end_char, current_label))
            
        training_data.append((text, {'entities': entities}))

    return training_data

def train_spacy_model(docs_df, n_iter=10, output_path='./models/spacy_finetuned'):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    for label in set(GT_LABEL_MAP.values()):
        if label != 'O':
            ner.add_label(label)

    training_data = prepare_spacy_training_data(docs_df)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        print("--- Starting spaCy Model Training ---")
        for it in range(n_iter):
            losses = {}
            random.shuffle(training_data)

            for text, annotations in tqdm(training_data, desc=f'Iteration {it+1}/{n_iter}'):
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example],drop=0.5, losses=losses)

    if output_path:
        nlp.to_disk(output_path)

    return nlp

def create_transformer_label_maps():
    gt_labels = set(GT_LABEL_MAP.values()) - {'O'}
    labels = ['O'] + [f'B-{t}' for t in gt_labels] + [f'I-{t}' for t in gt_labels]
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    return id2label, label2id

def prepare_transformer_training_data(docs_df, label2id):
    data = {'tokens': [], 'ner_tags': []}
    for doc_df in docs_df:
        words = doc_df['token'].tolist()
        tag_ids = [label2id[tag] for tag in doc_df['tag'].tolist()]

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
                label_ids.append(-100) # ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_transformer_model(
    docs_df,
    model_checkpoint='nlpaueb/legal-bert-base-uncased',
    n_epochs=3,
    output_dir='./transformer_finetuned_model'
):
    id2label, label2id = create_transformer_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    
    dataset = prepare_transformer_training_data(docs_df, label2id)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    print(f'--- Starting Transformer ({model_checkpoint}) Fine-Tuning ---')
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'Transformer model saved to {output_dir}')
    return model, tokenizer

def map_label(tag):
    if not isinstance(tag, str) or tag in ['O', '']:
        return 'O'
    
    parts = tag.split('-')
    prefix = parts[0]
    label_type = '-'.join(parts[1:]) if len(parts) > 1 else parts[0]
    
    mapped_type = GT_LABEL_MAP.get(label_type)
    
    if not mapped_type or mapped_type == 'O':
        return 'O'
    
    if prefix in ['B', 'I']:
        return f"{prefix}-{mapped_type}"
    else:
        return f"B-{mapped_type}"

def load_e_ner(path: str):
    df = pd.read_csv(path, names=['token', 'tag']).fillna('')
    df['tag'] = df['tag'].apply(map_label)
    
    split_idx = df.index[df['token'] == '-DOCSTART-'].tolist()
    
    # Create a list of DataFrames, one for each document
    list_dfs = [df.iloc[split_idx[i]+1:split_idx[i+1]].reset_index(drop=True) for i in range(len(split_idx)-1)]
        
    return list_dfs

if __name__ == '__main__':
    path = './data/E-NER-Dataset/all.csv'
    docs = load_e_ner(path)
    split_point = int(len(docs) * 0.8)
    train_docs = docs[:split_point]
    eval_docs = docs[split_point:]

    print(f"Loaded {len(docs)} documents.")
    print(f"Training set size: {len(train_docs)}")
    print(f"Evaluation set size: {len(eval_docs)}\n")
    
    # train spacy model
    """
    spacy_finetuned_path = './models/spacy_finetuned'
    train_spacy_model(train_docs, n_iter=300, output_path=spacy_finetuned_path)
    BENCHMARK_MODELS['spacy-finetuned'] = spacy_finetuned_path
    
    # fine-tune Legal-BERT model
    legalbert_finetuned_path = "./models/legalbert_finetuned"
    train_transformer_model(
        train_docs,
        model_checkpoint="nlpaueb/legal-bert-base-uncased",
        n_epochs=300,
        output_dir=legalbert_finetuned_path
    )
    BENCHMARK_MODELS['LegalBERT-Finetuned'] = legalbert_finetuned_path

    timestamp = datetime.now()
    timestamp = timestamp.strftime('%d-%m-%y_%H-%M-%S')
    with open('benchmark_checkpoint_'+timestamp+'.json', 'w') as f:
        json.dump(BENCHMARK_MODELS, f)
    """
    with open('benchmark_checkpoint_16-08-25_00-16-42.json') as f:
        BENCHMARK_MODELS = json.load(f)

    print(f"Evaluation set size: {len(eval_docs)}")
    benchmark_results = benchmark_models(eval_docs, BENCHMARK_MODELS)
    save_experiment_report(benchmark_results)
    plot_report(benchmark_results)
