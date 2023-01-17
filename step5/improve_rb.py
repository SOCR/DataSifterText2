from readability import evaluate_readability
from paraphrase import paraphrase
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import BertModel, BertConfig


def main(text, tokenizer, model):
    score_before = evaluate_readability(text)
    text_improved = paraphrase(tokenizer, model, text)
    score_after = evaluate_readability(text_improved)

    print(score_before)
    print(score_after)
    print(text_improved)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
    text = "The patient has been extubated. There is improvement in pulmonary edema."
    # main(sys.argv[1], tokenizer, model)
    main(text, tokenizer, model)