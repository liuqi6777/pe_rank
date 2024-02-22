import argparse
import json
import torch
from transformers import AutoTokenizer
from fastchat.model.model_adapter import get_conversation_template

from builder import load_pretrained_model


PLACEHOLDER = "<PLACEHOLDER>"
INSTRUCTION = """I will provide you with {num_retrieved_passages} passages, each indicated by a numerical identifier []. 
Rank the passages based on their relevance to the search query: {query}. 
{passages_llm_inputs}
Search Query: {query}. 
Rank the {num_retrieved_passages} passages above based on their relevance to the search query. 
All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.
"""


def eval_model(args):

    tokenizer, model, context_len = load_pretrained_model(
        model_path="./checkpoints/vicuna.jina.construct200k-rankgpt100k.finetune",
        model_base="lmsys/vicuna-7b-v1.5",
        model_name="ellama",
        projector_path="./checkpoints/vicuna.jina.construct200k.pretrain",
        use_flash_attn=True,
    )
    config = model.config
    model.to(torch.float16)
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    # load data
    data = []
    with open(args.input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    new_data = []
    for d in data:
        query = d["query"]
        retrieved_passages = d["hits"]
        num_retrieved_passages = len(retrieved_passages)
        passages_llm_inputs = "\n".join([f"[{i+1}] {PLACEHOLDER}" for i in range(num_retrieved_passages)])
        new_data.append({
            "conversations": [
                {
                    "from": "human",
                    "value": INSTRUCTION.format(
                        num_retrieved_passages=num_retrieved_passages,
                        query=PLACEHOLDER,
                        passages_llm_inputs=passages_llm_inputs
                    )
                },
            ],
            "extra_texts": [query] + [p["content"] for p in retrieved_passages] + [query]
        })
        
    # process data
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    extra_texts = []
    for i, source in enumerate(new_data):
        conversation = source["conversations"]
        if roles[conversation[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            conversation = conversation[1:]

        conv.messages = []
        for j, sentence in enumerate(conversation):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        
        extra_texts.append(source["extra_texts"])
        
    print(conversations[0])

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    
    extra_text_input_ids = []
    for texts in extra_texts:
        extra_text_input_ids.append(
            encoder_tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                max_length=200,
                truncation=True,
            ).input_ids
        )
    extra_text_input_ids = torch.nn.utils.rnn.pad_sequence(
        extra_text_input_ids,
        batch_first=True,
        padding_value=encoder_tokenizer.pad_token_id
    )
    

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            extra_text_input_ids=extra_text_input_ids,
            extra_text_attention_mask=extra_text_input_ids.ne(encoder_tokenizer.pad_token_id),
            do_sample=False,
            temperature=0.,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    stats_dict = {
        "ok": 0,
        "wrong_format": 0,
        "repetition": 0,
        "missing_documents": 0,
    }
    
    with open("results/output.txt", "w") as f:
        for output in outputs:
            f.write(output)
            f.write("\n")
    
    # format output
    results = []
    for i, response in enumerate(outputs):
        def _validate_format(response):
            for c in response:
                if not c.isdigit() and c != "[" and c != "]" and c != ">" and c != " ":
                    return False
            return True
        if not _validate_format(response):
            stats_dict["wrong_format"] += 1
            print(f"Wrong format: {response}")
            continue
        response = response[1:-1]
        if response[-1] == ']':
            response = response[:-1]
        ranks = response.split("] > [")
        try:
            tmp = []
            for rank in ranks:
                if int(rank) in tmp:
                    stats_dict["repetition"] += 1
                    continue
                if int(rank) >= len(data[i]["hits"]):
                    continue
                tmp.append(int(rank))
            ranks = tmp
        except ValueError:
            stats_dict["wrong_format"] += 1
            print(f"Wrong format: {response}")
            continue
        
        result = []
        hits = data[i]["hits"]
        # print(response)
        for j, rank in enumerate(ranks):
            result.append(f"{hits[0]['qid']} Q{i} {hits[rank-1]['docid']} {j} {1/(j+1)} rank")
        results.append(result)
        
    print(stats_dict)
    
    with open("results/test.txt", "w") as f:
        for result in results:
            f.write("\n".join(result))
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

