import torch
# import math
# import eng_to_ipa as ipa
import random


def fill_in_mask(tokenizer, model, sequence, mask='[MASK]', n=20):
    """Find the top n candidates of the masked word in sequence."""
    sequence = sequence.replace(mask, tokenizer.mask_token)

    inputs = tokenizer(sequence, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    token_logits = model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]

    top_tokens = torch.topk(mask_token_logits, n, dim=1).indices[0].tolist()
    top_tokens = [tokenizer.decode(token) for token in top_tokens
                  if '#' not in tokenizer.decode(token)]

    return top_tokens


# def best_rhyming_word(target, candidates, w1=1e-4, w2=0.2, w3=0.5, w4=0.5):
#     """Find the best rhyming word with target from the candidates.

#     w1 -- weight of penalty by priority order.
#     w2 -- weight of penalty by length difference.
#     w3 -- weight of penalty if same word.
#     w4 -- threshold score for replacement
#     """
#     target_ipa = ipa.convert(target)
#     best_score, best_candidate = w4, target
#     # print(f"{target:15} {target_ipa:>15}")
#     for i, candidate in enumerate(candidates):
#         candidate_ipa = ipa.convert(candidate)
#         length = 0
#         for ch1, ch2 in zip(target_ipa[::-1], candidate_ipa[::-1]):
#             if ch1 == ch2:
#                 length += 1
#             else:
#                 break
#         penalty1 = 1 - 0.6366 * math.atan(w1 * i)
#         penalty2 = 1 - 0.6366 * math.atan(w2 * abs(len(target_ipa)
#                                                    - len(candidate_ipa)))
#         penalty3 = w3 if candidate == target else 1
#         score = length * penalty1 * penalty2 * penalty3
#         # if length > 0:
#         #     print(f"{candidate:15} {candidate_ipa:>15}  {i:2}  {length}  {score:.4f}")
#         if score > best_score:
#             best_score, best_candidate = score, candidate
#     return best_candidate


# def rhyme_last_word(tokenizer, model, reference, source, pos):
#     """Rhyme the word at pos of source to that of reference."""
#     reference = reference.rstrip('.')
#     source = source.rstrip('.')
#     reference_word = reference.split(' ')[-pos]
#     source_words = source.split(' ')
#     source_words[-pos] = '[MASK]'
#     source = ' '.join(source_words) + '.'
#     candidates = fill_in_mask(tokenizer, model, source, n=200)
#     best_candidate = best_rhyming_word(reference_word, candidates)
#     result = source.replace('[MASK]', best_candidate)
#     return result


def paraphrase(tokenizer, model, sentence):
    words = sentence.rstrip('.').split(' ')
    indices = list(range(len(words)))
    random.shuffle(indices)
    for i in indices:
        words[i] = '[MASK]'
        replacement = fill_in_mask(tokenizer, model, ' '.join(words) + '.')[0]
        words[i] = replacement
        # print(' '.join(words) + '.')
    return ' '.join(words) + '.'