



def cot_infer(args, model, tokenizer, prompts, labels):

    full_predictions, short_predictions = [], []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        batch_outputs = model(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
        )
        batch_trajectories = tokenizer.batch_decode(batch_outputs)
        
        batch_prompts_concat = [
            f"""{input_}{trajectory} Therefore, the answer is """
            for input_, trajectory in zip(batch_prompts, batch_trajectories)
        ]
        tokenized_prompts_concat = tokenizer(batch_prompts_concat, padding="longest", return_tensors="pt")
        batch_outputs = model(
            tokenized_prompts_concat,
            max_new_tokens=1,
            do_sample=False,
            sequence_bias={tuple(tokenizer.encode(key)[-1:]): 100.0 for key in labels},
        )
        batch_full_text = tokenizer.batch_decode(batch_outputs)
        batch_full_predictions = [
            text[len(prompt):] for prompt, text in zip(batch_prompts, batch_full_text)
        ]
        batch_short_predictions = [
            text[len(prompt):] for prompt, text in zip(batch_prompts_concat, batch_full_text)
        ]
        full_predictions += batch_full_predictions
        short_predictions += batch_short_predictions
    
    return full_predictions, short_predictions

    





def main():
    dataset = ""
    model = ""
    tokenizer = ""
    prompts = func(dataset)
    targets = [sample["messages"["assistant"]["content"]] for sample in dataset]

    full_predictions, short_predictions = cot_infer(args, model, tokenizer, prompts, target)
    acc = exact_match.compute(predictions=short_predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]


