from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import time
import pandas as pd
import numpy as np

def train():
    # Chargement dataset
    dataset = pd.read_json('coordonnees_mots.json')

    # Chargement LLM (modèle de base et modèle PEFT)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
    )

    peft_model = get_peft_model(model, config)

    # Preprocessing
    # Pour chaque mot :
    def tokenize_function(mot):
        start_prompt = 'A partir des coordonnées de mains (n° du point de la main, x, y) fournies, donne le mot associé en langue des signes française. Fournis dans ta réponse le mot uniquement. \n\n'
        prompt = [start_prompt + input for input in mot]
        # Prompt
        mot['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True,
                                     return_tensors='pt').input_ids
        # Label : résultat à produite
        mot['labels'] = tokenizer(mot['mot'], padding='max_length', truncation=True,
                                  return_tensors='pt').input_ids

        return mot

    tokenize_datasets = dataset.map(tokenize_function, batched=True)
    tokenize_datasets = tokenize_datasets.remove_columns(['mot', 'id', 'x', 'y'])

    # Entraînement et sauvegarde
    output_dir = f'./dialogue-summary-training-{str(int(time.time()))}' #logs
    peft_training_args = TrainingArguments(output_dir=output_dir,
                                           auto_find_batch_size=True,
                                           learning_rate=1e-3,
                                           num_train_epochs=1,
                                           logging_steps=1,
                                           max_steps=1,
                                           report_to='none'
                                           )

    peft_trainer = Trainer(model=peft_model,
                           args=peft_training_args,
                           train_dataset=tokenize_datasets['train']
                           )


    peft_trainer.train()

    peft_model_path = './peft-dialogue-summary-checkpoint-local'

    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)

# Fonction qui génère depuis 
def generate(input):
    peft_model_base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')

    peft_model = PeftModel.from_pretrained(peft_model_base,
                                           './peft-dialogue-summary-checkpoint-local',
                                           torch_dtype=torch.bfloat16,
                                           is_trainable=False) ## is_trainable mean just a forward pass jsut to get a sumamry


    prompt = f"""
    A partir des coordonnées de mains (n° du point de la main, x, y) fournies, 
    donne le mot associé en langue des signes française. Fournis dans ta réponse le mot uniquement. \n

    {input}
    """

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    return tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

def main():
    train()
    #generate()

if __name__ == "__main__":
    main()