
import numpy as np
import os
from tqdm import tqdm
from retriever import Retriever
from typing import Optional
from openai import OpenAI

import argparse
import openai
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import utils as utils
import errno
import lm_dataformat

OPENAI_API_KEY = ''
client = OpenAI(api_key=OPENAI_API_KEY)

class LM:
    def get_perplexity_data(self, text) -> Optional[dict]:
        raise NotImplementedError

    @classmethod
    def create_from_config(cls, path):
        raise NotImplementedError

    def initialize_retriever(self, args):
        self.args = args
        if args.do_retrieval:
            self.retriever = Retriever(args)
        else:
            self.retriever = None

        
class GPT3LM(LM):

    def __init__(self, engine, save_dir_path, context_len=1024, max_seq_len=2048, verbose=False, batch_size=4, optimizer=None, args=None):
        self.engine = engine
        self.save_dir_path = save_dir_path
        self.context_len = context_len
        self.max_seq_len = max_seq_len
        self.wb = utils.WaitBlocker()
        self.verbose = verbose
        self.tmp = 1
        self.batch_size=batch_size
        self.optimzer=optimizer
        self.args = args
        self.training_step = 0

        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-xl')
        self.end_of_text_token_id = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]


    def _initial_optimizer(self):
        self.optimizer = optim.AdamW(self.retriever.model.parameters(), lr=1e-5)


    def forward_training(self, text):
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        batch_loss = []
        batch_index = 0
        step = 0
        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            query_id = input_tokens[:-len(pred_tokens)]
            query = self.tokenizer.decode(query_id)
            if query_id  == [] or query not in self.retriever.query2docs:
                continue
            retriever_loss = self.forward_training_single(input_tokens, pred_tokens)
            retriever_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # print(f'retriever_loss {retriever_loss.item()} \t step {self.training_step}.')
            self.training_step += 1


    def forward_training_single(self, input_tokens, pred_tokens):  
        query_id = input_tokens[:-len(pred_tokens)]
        # print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))
        query = self.tokenizer.decode(query_id)

        if query_id != []:
            docs, scores = self.retriever.retrieve_passage([query])
            plain_docs = [doc["text"] for doc in docs]

            # encode the retrieved docs
            query_emb = self.retriever.embed_queries([query])
            # query_emb = self.retriever([query])
            passage_emb = self.retriever.embed_queries(plain_docs)
            # passage_emb = self.retriever(plain_docs).to(query_emb)
            passage_emb = passage_emb.view(1, -1, passage_emb.size(-1))
            retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
            all_gold_score = []
            for i in range(len(docs)):
                doc_str = plain_docs[i]
                doc_encodings = self.tokenizer.encode(doc_str)
                input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
                block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens,)
                gold_score = block_output["logprobs"]
                # gold_score = None
                while type(gold_score) is not np.ndarray:
                    block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens,)
                    gold_score = block_output["logprobs"]
                all_gold_score.append(gold_score.mean())
            all_gold_score = torch.FloatTensor(all_gold_score).unsqueeze(0)
            retriever_loss = self.kldivloss(retriever_score, all_gold_score)
        else:
            raise ValueError('input_ids invalid')

        return retriever_loss

    
    def save_model(self, step):

        def symlink_force(target, link_name):
            try:
                os.symlink(target, link_name)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(link_name)
                    os.symlink(target, link_name)
                else:
                    raise e
        model_to_save = self.retriever.model
        path = os.path.join(self.save_dir_path, "checkpoint")
        model_name = "step-" + str(step)
        epoch_path = os.path.join(path, model_name)  # "step-%s" % step)
        os.makedirs(epoch_path, exist_ok=True)
        cp = os.path.join(path, "latest")
        fp = os.path.join(epoch_path, "checkpoint.pth")

        checkpoint = {
        "step": step,
        "model": model_to_save.state_dict(),
    }
        torch.save(checkpoint, fp)
        symlink_force(epoch_path, cp)

          
    
    def kldivloss(self, score, gold_score):
        # gold_score = torch.Tensor(gold_score).to(score)
        gold_score = torch.softmax(gold_score / self.args.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.args.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)
    
    # noinspection DuplicatedCode
    def get_perplexity_data(self, text) -> Optional[dict]:
        input_ids = self.tokenizer.encode_plus(text)["input_ids"]
        rolling_token_windows = utils.get_rolling_token_windows(
            token_list=input_ids,
            prefix_token=self.end_of_text_token_id,
            max_seq_len=self.max_seq_len,
            context_len=self.context_len,
        )

        # noinspection PyListCreation
        all_logprobs = []
        all_positions = []

        # Remaining windows: input_tokens are context, pred_tokens are prediction
        for input_tokens, pred_tokens in tqdm(rolling_token_windows):
            # ipdb.set_trace()
            # assert len(input_tokens) == 256
            # assert len(pred_tokens) == 512
            # bp()
            query_id = input_tokens[:-len(pred_tokens)]
            print("len(context):", len(query_id), "len(pred_tokens):", len(pred_tokens))
            # do retrieval
            if self.args.do_retrieval and (query_id != []):
                if self.args.random == 0:
                    query = self.tokenizer.decode(query_id)
                else:
                    query = "who is US president?"
                docs, scores = self.retriever.retrieve_passage([query])
                plain_docs = [doc["text"] for doc in docs]

                if self.args.ensemble == 0:
                    doc_str = "\n".join(plain_docs)
                    print(f"query: {[query]}\nretrieved doc: {[doc_str]}")
                    doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                    input_tokens = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
                    print("retrieve + context: ", len(input_tokens)-len(pred_tokens))
                else:
                    '''
                    a + b + c = log(e^log(a) + e^log(b) + e^log(c))
                    '''
                    logprobs_list = []
                    block_output = None
                    assert self.args.ensemble <= len(plain_docs)
                    
                    for i in range(self.args.ensemble):
                        doc_str = plain_docs[i]
                        doc_encodings = self.tokenizer.encode(doc_str)[:self.args.retrieved_max_length]
                        input_tokens_tmp = torch.concat((torch.LongTensor(doc_encodings), torch.LongTensor(input_tokens)), dim=-1)
                        block_output = self.get_token_logprobs(input_tokens=input_tokens_tmp, pred_tokens=pred_tokens,)
                        logprobs_list.append(block_output["logprobs"])
                        # sum(np.isinf(block_output["logprobs"]))
                    # bp()
                    # block_output["logprobs"] = np.log(np.mean(np.exp(logprobs_list), axis=0))
                    block_output["logprobs"] = torch.logsumexp(torch.FloatTensor(logprobs_list), dim=0) - np.log(len(logprobs_list))
                    block_output["logprobs"] = block_output["logprobs"].numpy()
            else:
                # bp()
                block_output = self.get_token_logprobs(input_tokens=input_tokens, pred_tokens=pred_tokens,)
            # bp()
            all_logprobs.append(block_output["logprobs"])
            all_positions.append(block_output["positions"])

        if not all_logprobs:
            return None

        # Gather
        all_logprobs = np.concatenate(all_logprobs)
        all_positions = np.concatenate(all_positions)
        assert len(all_logprobs) == len(input_ids)
        return {
            "logprobs": all_logprobs,
            "positions": all_positions,
            "length": len(all_logprobs),
            "utf8_length": len(text.encode('utf-8')),
        }

    def get_token_logprobs(self, input_tokens, pred_tokens):
        pred_start = len(input_tokens) - len(pred_tokens) + 1
        # We're going to stitch together the input_tokens and pred_tokens
        # In the longest case, this gets us to length = max_seq_len+1 (which the API works with)
        input_tokens = np.array(input_tokens).tolist()
        assert input_tokens[pred_start:] == pred_tokens[:-1]
        token_ids = input_tokens + [pred_tokens[-1]]
        prompt_string = self.tokenizer.decode(token_ids)
        try:
            with self.wb.check_valid():
                # response = openai.Completion.create(
                #     engine=self.engine,
                #     prompt=token_ids,
                #     max_tokens=0,
                #     temperature=0.0,
                #     logprobs=0,
                #     echo=True,
                # )
                response = client.chat.completions.create(
                    model=self.engine,
                    messages=[
                        {"role": "user", "content": prompt_string},
                    ],
                    logprobs=True,
                )
                # logprobs = np.array(response["choices"][0]["logprobs"]["token_logprobs"][pred_start:])
                # logprobs = response.choices[0].logprobs.content[pred_start:]
                logprobs_list = []
                for i in range(len(token_ids[pred_start:])):
                    logprobs_item = response.choices[0].logprobs.content[i].logprob
                    logprobs_list.append(logprobs_item)
                logprobs = np.array(logprobs_list)
                if self.verbose:
                    print("Context:", self.tokenizer.convert_ids_to_tokens(token_ids))
                    print("Predicting:", self.tokenizer.convert_ids_to_tokens(token_ids)[pred_start:])
                    print("Perplexity:", np.exp(-logprobs.mean()))
                    print()

                positions = np.arange(pred_start-1, pred_start-1 + len(token_ids[pred_start:]))
            
        except:
            logprobs = None
            positions = None

        return {
            "logprobs": logprobs,
            "positions": positions,
        }

    @classmethod
    def create_from_config(cls, config):
        return cls(**config)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', type=str, default="preset_configs/bloom-3b.json")
    parser.add_argument('--data',
                        default="wikitext-2-v1", type=str)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--output_path', type=str, default='outputs/ppl.data')
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--doc_indices_path', type=str, default=None)
    parser.add_argument('--per_gpu_batch_size', type=int, default=64)
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")


    # retrieval
    parser.add_argument('--do_retrieval', type=int, default=0,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--use-faiss-gpu', action="store_true", 
                        help='If enabled, use faiss GPU for retrieval inference')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help='Number of GPU used in retrieval')
    parser.add_argument('--ensemble', type=int, default=5,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--passages', type=str, default="./data/text.jsonl",
                        help='Path to passages (.tsv file)')  # wikitext-103-v1, wikitext-2-v1
    parser.add_argument('--passages_embeddings', type=str,
                        default="./data/embeddings/*", help='Glob path to encoded passages')
    parser.add_argument('--n_docs', type=int, default=5,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--chunk_size', type=int, default=64,
                        help="Maximum number of words in a chunk")
    parser.add_argument('--normalize_text',
                        action='store_true', help="normalize text")
    parser.add_argument('--question_maxlength', type=int, default=128, help="Maximum number of tokens in a question")
    parser.add_argument('--random', type=int, default=0, help="random document")
    parser.add_argument('--temperature_gold', type=int, default=0.1, help="temperature gold score")
    parser.add_argument('--temperature_score', type=int, default=0.1, help="temperature retriever score")


    # 1024:
    parser.add_argument('--retrieved_max_length', type=int, default=128)
    parser.add_argument('--context_len', type=int, default=2)
    parser.add_argument('--pred_len', type=int, default=4)

    parser.add_argument('--re_model_name_or_path', type=str, default="facebook/contriever",
                        help="path to directory containing model weights and config file")

    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument("--n_subquantizers", type=int, default=0,
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n_bits", type=int, default=8,
                        help='Number of bits per subquantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=1000000,
                        help="Batch size of the number of passages indexed")
    parser.add_argument("--save_or_load_index", action='store_true',
                        help='If enabled, save index and load index if it exists')
    parser.add_argument("--cache_dict", type=str, default='query2doc.json', 
                        help='Retrieved query2docs filename')
    return parser.parse_args()


def create_model(json_path):
    config = utils.read_json(json_path)
    model_type = config.pop("model_type")
    model_name = config.pop("model_name")
    device = config.pop("device")
    model = GPT3LM.create_from_config(config)
    return model


def model_train(model, indices=None, args=None):
    reader = lm_dataformat.Reader(args.data)
    # embed()  # set ratio
    # for i, doc in enumerate(tqdm_lib.tqdm(reader.stream_data())):
    for i, doc in tqdm(enumerate(reader.stream_data)):
        if indices is not None and i not in indices:
            continue
        model.forward_training(doc)
    
    model.save_model(step=model.training_step)
        
    


def main():
    args = parse_args()
    model = create_model(args.model_config_path)
    model.context_len = args.context_len
    model.max_seq_len = args.context_len + args.pred_len
    if args.retrieved_max_length != 0:
        args.do_retrieval=1
    else:
        args.do_retrieval=0
    model.initialize_retriever(args)
    model._initial_optimizer()

    if args.doc_indices_path:
        assert args.max_docs is None
        indices = set(utils.read_json(args.doc_indices_path))
    elif args.max_docs:
        assert args.doc_indices_path is None
        indices = set(range(args.max_docs))
    else:
        indices = None

    model_train(model=model, args=args)


if __name__ == '__main__':
    main()

    
