# AI Collaboration MD

### - Tool: Claude Code, model: Claude Opus 5.5

### - Session: 8 Nov 2026

### - Purpose: Stage 2 of AI Improvement, changes include word dropout, and trainable <UNK>

▐▛███▛█ Claude Code v2.1.316
▝▜██████▀ Opus 5.5 · Claude Pro
▝▝ ▝▝ D:\Coding\Encoder-Only-Architect

⚠ Your login expires in 3 days · run /login to renew

❯ Hi claude, okay so I want to improve on the training for this model.  
 Currently the baseline model gets intent 82.25% and slot F1 58.19%. The slot  
 accuracy is only accurate slightly more than half the time. Can I see whats  
 the weak point of the training pipeline and how to improve. Just discuss  
 first for now before modifying the code.

some pointers:

- do not edit the baseline code, only modify the function afterwards so there
  is a record of early + improvement for the model
- do not modify/edit the database or the split, its already a given
- only train on train split and tune on dev split, test is just for  
  evaluation later on.
- lastly ensure whoel model is under 20M parameters

Read 2 files, listed 1 directory, ran 2 shell commands (ctrl+o to expand)
⎿  Interrupted · What should Claude do instead?

✻ Cooked for 39s · done 8:06 PM

❯ ok wait, the main training code is in main.ipynb

● I've already read main.ipynb. That command was only going to count how many
dev/test words never appear in training. You asked for discussion only, so
here it is with no more tool calls and no code changes.

Weak points in the current pipeline

1. Heavy overfitting with nothing to regularise it (biggest issue)

From your training log:

- Train loss falls from 2.97 to 0.03.
- Dev loss is lowest at epoch 4 (1.05) and then climbs to 2.05.
- Dev slot F1 levels off around 61–64% from epoch 5 onward.

Training has no dropout, no weight decay, no learning-rate schedule and no
gradient clipping. About 530k of the 624k parameters are the trainable
embedding table (5294 × 100). The model mostly memorises training words rather
than learning patterns.

2. The <UNK> vector is never trained (likely the main reason slot F1 is low)

The vocabulary is built from train only, so during training no token is ever
<UNK>. Its vector stays at the random value it started with. At dev/test time,
every unseen word (person names, song names, places, apps) maps to that
untrained vector, and these are exactly the words slots are made of. This
probably also explains why test slot F1 (58%) is so far below dev (64%).

3. Fine-tuned GloVe drifts, and test words GloVe already knows get thrown away

- A word in dev/test but not in train becomes <UNK>, even when GloVe has a
  good vector for it.
- Fine-tuning moves the vectors of training words, so even if you added those
  GloVe words back, they would no longer sit in the same space as the training
  words.

4. Slot decoding ignores BIO rules

Each token takes its own argmax over 111 tags, so the model can output
sequences like O I-time. decode_slot_pairs patches these, but the metric is
exact span match, so a single wrong boundary tag costs a false positive and a
false negative.

5. The slot head has very little to work with

- One layer, one head, d_model=100, and a direct linear layer to 111 tags.
- No character or subword information, so the model can't tell from spelling
  that "tulsa" looks like a place.
- The slot head never sees the predicted intent, even though intent strongly
  limits which slots can appear (alarm_set → time/date).

6. Smaller issues

- Loss imbalance: the slot loss is averaged over tokens, most of which are O.
  Intent and slot losses are just added with weight 1:1.
- Checkpoint selection is noisy: combined dev scores range only 72.4–73.7
  across epochs, and results come from one seed. The gains I suggest are a few
  points each, so they should be compared as a mean over 3 seeds.
- Positional embeddings: 64 learned positions but an average length of 7, so
  positions beyond about 20 are barely trained (minor).

Suggested improvements, in priority order

#: 1
Change: Word dropout: during training, randomly replace tokens with <UNK>
(e.g.
p=0.1, or words seen only once)
Targets: #2
Extra params: 0
────────────────────────────────────────
#: 2
Change: Dropout (0.1–0.3) on embeddings, attention weights and FFN, plus AdamW

with weight decay, warmup + cosine/linear decay, gradient clipping, and early
stopping on dev
Targets: #1
Extra params: 0
────────────────────────────────────────
#: 3
Change: Freeze or partly freeze GloVe, e.g. frozen GloVe plus a small
trainable
projection or adapter. Optionally fill the lookup table from GloVe's own
vocabulary (e.g. its top 100k words, about 10M params) so unseen words get
real vectors
Targets: #2, #3
Extra params: up to ~10M
────────────────────────────────────────
#: 4
Change: CRF layer or constrained Viterbi decoding for BIO tags
Targets: #4
Extra params: ~12k (CRF) / 0
────────────────────────────────────────
#: 5
Change: Character CNN embedding concatenated with GloVe
Targets: #5
Extra params: ~50–100k
────────────────────────────────────────
#: 6
Change: Intent-to-slot conditioning: concatenate the pooled or intent
representation onto each token before the slot head
Targets: #5
Extra params: small
────────────────────────────────────────
#: 7
Change: Wider or deeper encoder: project to d=256, 2–4 layers, multiple heads;

or switch to GloVe 300d
Targets: #5
Extra params: 1–5M
────────────────────────────────────────
#: 8
Change: Weighted joint loss, label smoothing on intent
Targets: #6
Extra params: 0

The cheapest, highest-value first round is 1 + 2 + 4. It adds no new data and
almost no parameters, and I'd expect a clear jump in slot F1. Items 3, 5 and 6
can come next, measured one at a time on dev so each gain is attributable.

How I'd keep your rules

- Baseline left alone: all baseline cells stay untouched. New cells go below
  the baseline under a new section such as Part D: Improved Model, with a new
  model class, new training function and new checkpoint file, so baseline and
  improved results sit side by side.
- Splits and data: the same data and partitions are reused unchanged. The
  vocabulary is built from train only, and GloVe is external pretrained data,
  so expanding the lookup from it doesn't touch dev/test.
- Tuning: all tuning and early stopping use dev only. Test is evaluated once
  at the end.
- 20M limit: keep the assert total_params <= 20_000_000 check, counting frozen
  embeddings too.

Two questions before I write code

1. Architecture: does "Stage 1" require keeping one encoder layer and one
   attention head? If yes, I'll stick to items 1–6 and 8. If no, item 7 is
   available too.
2. Expanding the vocabulary: is filling the embedding lookup from GloVe's own
   word list allowed in your course? It's a standard, legitimate technique,
   but some assignments limit the vocabulary to train words only.

✻ Churned for 22s · done 8:06 PM

※ recap: You want better slot F1 than the baseline without touching its code, and I've only proposed improvements so far, starting with word dropout, regularisation and CRF decoding. Next, I need your answers on keeping one layer and one head, and on whether filling the vocabulary from GloVe is allowed. (disable recaps in /config)

❯ ok lets try two improvement first. 1 is the UNK seems like biggest issue,  
 lets go with the approach of words seen below a threshold, maybe less than 1  
 or 2. another improvement is to your recommendation number 2, which is  
 dropout, attention weights and FFN, adamW so that the training can have no  
 memorization and no overfitting later on.

try this first.

keep your recommended change 3 on the reserve for now. its going to get  
 implemented after we full run the changes above.

● Edit Notebook(main.ipynb@fb25c99b)
⎿  Updated cell 365bf1bb:
1 # Part D. Improvement 1: Rare-word UNK replacement + Regulariz
ation #
2
3 The baseline (Parts A–C) is left untouched so its results rema
in on record. This part only adds new cells.
4
5 ### Weak points addressed ###
6 1. **`<UNK>` was never trained.** The vocabulary is built from
the train split, so during training no token is ever `<UNK>`.
Its embedding stays random, yet every unseen dev/test word (n
ames, songs, places, the words slots are made of) maps to it.
7 - **Fix (word dropout):** during training only, each token
whose train frequency is `<= RARE_THRESHOLD` is replaced by `<
           UNK>` with probability `UNK_REPLACE_PROB`. The model learns a
useful `<UNK>` vector and learns to tag slots from context.
8 2. **Heavy overfitting.** In the baseline, train loss reaches
0.03 while dev loss rises from 1.05 (epoch 4) to 2.05, with no
regularization anywhere.
9 - **Fix:** dropout on embeddings, attention weights, the FF
N and residual branches, and before both classifier heads. **A
damW** with decoupled weight decay (not applied to embeddings,
biases or LayerNorm), linear warmup + cosine LR decay, gradie
nt clipping and early stopping on DEV.
10
11 The architecture is unchanged: 1 encoder layer, 1 self-attenti
on head, GloVe 100d, same parameter count. Training uses the t
rain split only, tuning/selection uses DEV only, and TEST is e
valuated once at the end.

● Edit Notebook(main.ipynb@365bf1bb)
⎿  Updated cell fcd86ffe:
1 ## D1. Rare-word `<UNK>` replacement (training only) ##
2
3 `word_counts` comes from the train split (A5), so the rare-word
set is built from train data only. The replacement happens ins
ide a new training collate function. DEV and TEST loaders keep
using the original `collate_fn`, so evaluation is unaffected.
4
5 - `RARE_THRESHOLD = 1`: words seen once in train are candidates
. Set it to 2 to include words seen twice.
6 - `UNK_REPLACE_PROB = 0.5`: each time a rare token appears in a
batch, it is replaced with probability 0.5, so the model still
sees the real word half the time.

● Edit Notebook(main.ipynb@fcd86ffe)
⎿  Updated cell 1f7c548b:
1 SEED*V2 = 42
2 RARE_THRESHOLD = 1 # words with train count <= this are
eligible for <UNK> replacement
3 UNK_REPLACE_PROB = 0.5 # probability of replacing an eligib
le token in each batch
4
5 # Rare-word IDs, built from TRAIN counts only (word_counts fro
m A5)
6 rare_word_ids = sorted(vocab[w] for w, c in word_counts.items(
) if c <= RARE_THRESHOLD)
7 rare_mask_lookup = torch.zeros(len(vocab), dtype=torch.bool)
8 rare_mask_lookup[rare_word_ids] = True
9
10 train_token_total = sum(word_counts.values())
11 rare_token_total = sum(c for c in word_counts.values() if c <=
RARE_THRESHOLD)
12 print(f"Rare word types (count <= {RARE_THRESHOLD}): {len(rare
\_word_ids):,} / {len(word_counts):,} "
13 f"({100 * len(rare*word_ids) / len(word_counts):.1f}% of
vocab)")
14 print(f"Rare tokens: {rare_token_total:,} / {train_token_total
:,} "
15 f"({100 * rare*token_total / train_token_total:.2f}% of
train tokens)")
16 print(f"Expected <UNK> rate during training: {100 * UNK*REPLAC
E_PROB * rare_token_total / train_token_total:.2f}% of tokens"
)
17
18 unk_generator = torch.Generator()
19 unk_generator.manual_seed(SEED_V2)
20
21 def collate_fn_word_dropout(batch):
22 """Training-only collate: same padding as collate_fn, then
rare tokens -> <UNK> with prob UNK_REPLACE_PROB."""
23 out = collate_fn(batch)
24 input_ids = out["input_ids"].clone()
25 eligible = rare_mask_lookup[input_ids] & out["attention_ma
sk"]
26 coin = torch.rand(input_ids.shape, generator=unk_generator
) < UNK_REPLACE_PROB
27 input_ids[eligible & coin] = vocab["<UNK>"]
28 out["input_ids"] = input_ids # slot/intent labels are unc
hanged
29 return out
30
31 # Sanity check on one batch
32 \_check = collate_fn_word_dropout([train_dataset[i] for i in ra
nge(64)])
33 print("UNK tokens in sample batch:",
34 int((\_check["input_ids"] == vocab["<UNK>"]).sum()), "of"
, int(\_check["attention_mask"].sum()))

● Edit Notebook(main.ipynb@1f7c548b)
⎿  Updated cell 0d87cd50:
1 ## D2. Regularized Encoder-only Model ##
2
3 Same pipeline as B1 (GloVe + learned positions → 1 single-hea
d self-attention → Add & Norm → FFN → Add & Norm → intent / s
lot heads), with dropout added at these points:
4
5 | Location | Rate |
6 |---|---|
7 | Word + position embeddings | `EMB_DROPOUT` |
8 | Attention weights (after softmax) | `ATTN_DROPOUT` |
9 | Inside FFN (after ReLU) | `HIDDEN_DROPOUT` |
10 | Attention and FFN outputs before each residual Add | `HIDDE
           N_DROPOUT` |
11 | Before intent and slot classifiers | `HIDDEN_DROPOUT` |
12
13 Dropout has no parameters, so the parameter count matches the
baseline. It is disabled automatically in `model.eval()`, wh
ich `evaluate()` already calls.

● Adding the new model class cell, then the training setup and loop.

● Edit Notebook(main.ipynb@0d87cd50)
⎿  Updated cell ae0c182e:
1 class RegularizedSingleHeadEncoder(nn.Module):
2 def **init**(self, glove*vectors, pad_id, max_positions, n
um_intents, num_bio_tags,
3 ffn_dim=200, emb_dropout=0.2, attn_dropout=0.
1, hidden_dropout=0.2):
4 super().**init**()
5 vectors = torch.as_tensor(glove_vectors, dtype=torch.f
loat32).clone()
6 d_model = vectors.size(1)
7 self.d_model = d_model
8 self.max_positions = max_positions
9
10 # Word vectors: initialized with GloVe and trainable (
same as baseline)
11 self.word_embeddings = nn.Embedding.from_pretrained(ve
ctors, freeze=False, padding_idx=pad_id)
12
13 # Learned positional embeddings
14 self.position_embeddings = nn.Embedding(max_positions,
d_model)
15 nn.init.normal*(self.position_embeddings.weight, mean=
0.0, std=0.02)
16
17 # Single-head attention parameters
18 self.W_q = nn.Linear(d_model, d_model)
19 self.W_k = nn.Linear(d_model, d_model)
20 self.W_v = nn.Linear(d_model, d_model)
21
22 # Dropout layers (no parameters)
23 self.emb_dropout = nn.Dropout(emb_dropout)
24 self.attn_dropout = nn.Dropout(attn_dropout)
25 self.hidden_dropout = nn.Dropout(hidden_dropout)
26
27 # Add & Norm + FFN + Add & Norm
28 self.norm1 = nn.LayerNorm(d_model)
29 self.ffn = nn.Sequential(
30 nn.Linear(d_model, ffn_dim),
31 nn.ReLU(),
32 nn.Dropout(hidden_dropout),
33 nn.Linear(ffn_dim, d_model)
34 )
35 self.norm2 = nn.LayerNorm(d_model)
36
37 # Two heads sharing the encoder output
38 self.intent_classifier = nn.Linear(d_model, num_intent
s)
39 self.slot_classifier = nn.Linear(d_model, num_bio_tags
)
40
41 def forward(self, input_ids, attention_mask):
42 batch_size, seq_len = input_ids.shape
43 if seq_len > self.max_positions:
44 raise ValueError(f"Sequence length {seq_len} excee
ds maximum {self.max_positions}")
45 attention_mask = attention_mask.bool()
46 position = torch.arange(seq_len, device=input_ids.devi
ce).unsqueeze(0).expand(batch_size, -1)
47
48 # Embeddings + embedding dropout
49 X = self.emb_dropout(self.word_embeddings(input_ids) +
self.position_embeddings(position))
50
51 # Single-head scaled dot-product attention with paddin
g mask
52 Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
53 scores = torch.matmul(Q, K.transpose(-2, -1)) / math.s
qrt(self.d_model)
54 scores = scores.masked_fill(~attention_mask.unsqueeze(
1), float('-inf'))
55 weights = self.attn_dropout(torch.softmax(scores, dim=
-1))
56 attended = torch.matmul(weights, V)
57
58 # Add & Norm -> FFN -> Add & Norm (dropout on each res
idual branch)
59 H1 = self.norm1(X + self.hidden_dropout(attended))
60 H = self.norm2(H1 + self.hidden_dropout(self.ffn(H1)))
61
62 # Intent: masked mean pooling
63 valid = attention_mask.unsqueeze(-1).to(dtype=H.dtype)
64 pooled = (H \* valid).sum(dim=1) / valid.sum(dim=1).cla
mp(min=1.0)
65 intent_logits = self.intent_classifier(self.hidden_dro
pout(pooled))
66
67 # Slot: per-token classification
68 slot_logits = self.slot_classifier(self.hidden_dropout
(H))
69 return intent_logits, slot_logits

● Edit Notebook(main.ipynb@ae0c182e)
⎿  Updated cell e96ed8c5:
1 ## D3. Training Setup (AdamW + warmup/cosine schedule) ##
2
3 ### Configurations: ###
4 - Seed: 42 (reset so the comparison with the baseline is fair)
5 - Max epochs: 40, early stopping with patience 8 on the DEV se
lection score
6 - Learning rate: 1e-3, 1 epoch linear warmup, then cosine deca
y to 0
7 - Batch size: 32
8 - AdamW weight decay: 0.01 on Linear weights only. Embeddings,
biases and LayerNorm are excluded, because decay would shrink
GloVe vectors of words that rarely receive gradients.
9 - Gradient clipping: max norm 1.0
10 - Dropout: embeddings 0.2, attention 0.1, hidden/FFN/heads 0.2
11
12 The baseline objects (`model`, `optimizer`, `train_loader`, `h
           istory`) are not overwritten. Everything here uses a `_v2` suf
fix.

● Edit Notebook(main.ipynb@PLACEHOLDER)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@e96ed8c5)
⎿  Updated cell e3692a04:
1 random.seed(SEED*V2)
2 np.random.seed(SEED_V2)
3 torch.manual_seed(SEED_V2)
4
5 NUM_EPOCHS_V2 = 40
6 PATIENCE_V2 = 8
7 LR_V2 = 1e-3
8 WEIGHT_DECAY_V2 = 0.01
9 WARMUP_EPOCHS_V2 = 1
10 GRAD_CLIP_V2 = 1.0
11 EMB_DROPOUT = 0.2
12 ATTN_DROPOUT = 0.1
13 HIDDEN_DROPOUT = 0.2
14 CHECKPOINT_PATH_V2 = "model_checkpoint_v2.pt"
15
16 shuffle_generator_v2 = torch.Generator()
17 shuffle_generator_v2.manual_seed(SEED_V2)
18 unk_generator.manual_seed(SEED_V2) # reset after the D1 sanit
y check
19
20 # Training loader uses rare-word <UNK> replacement; DEV/TEST l
oaders are unchanged
21 train_loader_v2 = DataLoader(
22 train_dataset,
23 batch_size=BATCH_SIZE,
24 shuffle=True,
25 collate_fn=collate_fn_word_dropout,
26 num_workers=0,
27 generator=shuffle_generator_v2
28 )
29
30 model_v2 = RegularizedSingleHeadEncoder(
31 glove_vectors=embedding_matrix,
32 pad_id=vocab["<PAD>"],
33 max_positions=max_length,
34 num_intents=len(intent_to_id),
35 num_bio_tags=len(tag_to_id),
36 ffn_dim=200,
37 emb_dropout=EMB_DROPOUT,
38 attn_dropout=ATTN_DROPOUT,
39 hidden_dropout=HIDDEN_DROPOUT,
40 )
41
42 total_params_v2 = sum(p.numel() for p in model_v2.parameters()
)
43 trainable_params_v2 = sum(p.numel() for p in model_v2.paramete
rs() if p.requires_grad)
44 assert total_params_v2 <= 20_000_000
45
46 # AdamW parameter groups: decay only Linear weight matrices
47 decay, no_decay = [], []
48 for name, param in model_v2.named_parameters():
49 if not param.requires_grad:
50 continue
51 if param.ndim >= 2 and "embeddings" not in name:
52 decay.append(param)
53 else:
54 no_decay.append(param) # embeddings, biases, LayerNor
m
55
56 optimizer_v2 = torch.optim.AdamW(
57 [
58 {"params": decay, "weight_decay": WEIGHT_DECAY_V2},
59 {"params": no_decay, "weight_decay": 0.0},
60 ],
61 lr=LR_V2,
62 )
63
64 # Linear warmup then cosine decay, stepped once per batch
65 steps_per_epoch = len(train_loader_v2)
66 total_steps = NUM_EPOCHS_V2 * steps*per_epoch
67 warmup_steps = WARMUP_EPOCHS_V2 * steps*per_epoch
68
69 def lr_lambda(step):
70 if step < warmup_steps:
71 return (step + 1) / warmup_steps
72 progress = (step - warmup_steps) / max(1, total_steps - wa
rmup_steps)
73 return 0.5 * (1.0 + math.cos(math.pi \_ min(progress, 1.0))
)
74
75 scheduler_v2 = torch.optim.lr_scheduler.LambdaLR(optimizer_v2,
lr_lambda)
76
77 print("Seed:", SEED_V2)
78 print(f"Max epochs: {NUM_EPOCHS_V2} (patience {PATIENCE_V2}),
batch size: {BATCH_SIZE}, peak LR: {LR_V2}")
79 print(f"AdamW weight decay: {WEIGHT_DECAY_V2} on {sum(p.numel(
) for p in decay):,} params, "
80 f"none on {sum(p.numel() for p in no_decay):,} params")
81 print(f"Dropout: emb {EMB_DROPOUT}, attn {ATTN_DROPOUT}, hidde
n {HIDDEN_DROPOUT}")
82 print(f"Rare-word UNK: count <= {RARE_THRESHOLD}, p = {UNK_REP
LACE_PROB}")
83 print(f"Total parameters: {total_params_v2:,} | Trainable: {tr
ainable_params_v2:,}")

● Edit Notebook(main.ipynb@e3692a04)
⎿  Updated cell b4d41494:
1 ## D4. Training Loop with Early Stopping ##
2
3 Same selection score as the baseline (mean of DEV intent accura
cy and DEV slot micro-F1). Training stops if it doesn't improve
for `PATIENCE_V2` epochs. The current LR is logged each epoch.

● Edit Notebook(main.ipynb@b4d41494)
⎿  Updated cell 4ccff27e:
1 history*v2 = []
2 best_score_v2 = float('-inf')
3 best_epoch_v2 = None
4 epochs_without_improvement = 0
5
6 for epoch in range(1, NUM_EPOCHS_V2 + 1):
7 start = time.perf_counter()
8 model_v2.train() # enables dropout (evaluate() switches
back to eval mode)
9
10 total_train = intent_train = slot_train = 0.0
11 train_examples = 0
12
13 for batch in train_loader_v2:
14 input_ids = batch["input_ids"]
15 attention_mask = batch["attention_mask"]
16 intent_targets = batch["intent_ids"]
17 slot_targets = batch["slot_ids"]
18
19 optimizer_v2.zero_grad(set_to_none=True)
20 intent_logits, slot_logits = model_v2(input_ids, atte
ntion_mask)
21 loss, intent_loss, slot_loss = compute_loss(
22 intent_logits, slot_logits, intent_targets, slot*
targets
23 )
24 loss.backward()
25 torch.nn.utils.clip*grad_norm*(model*v2.parameters(),
GRAD_CLIP_V2)
26 optimizer_v2.step()
27 scheduler_v2.step()
28
29 n = input_ids.size(0)
30 train_examples += n
31 total_train += loss.item() * n
32 intent*train += intent_loss.item() * n
33 slot_train += slot_loss.item() \* n
34
35 # DEV only: no parameter updates
36 dev_result = evaluate(model_v2, dev_loader)
37 selection_score = (dev_result["intent_accuracy"] + dev_re
sult["slot_micro_f1"]) / 2.0
38 elapsed = time.perf_counter() - start
39 current_lr = scheduler_v2.get_last_lr()[0]
40
41 record = {
42 "epoch": epoch,
43 "lr": current_lr,
44 "train_loss": total_train / train_examples,
45 "train_intent_loss": intent_train / train_examples,
46 "train_slot_loss": slot_train / train_examples,
47 "val_loss": dev_result["loss"],
48 "val_intent_loss": dev_result["intent_loss"],
49 "val_slot_loss": dev_result["slot_loss"],
50 "val_intent_accuracy": dev_result["intent_accuracy"],
51 "val_slot_micro_f1": dev_result["slot_micro_f1"],
52 "selection_score": selection_score,
53 "elapsed_time_sec": elapsed
54 }
55 history_v2.append(record)
56
57 print(
58 f"Epoch {epoch:02d}/{NUM_EPOCHS_V2} | lr {current_lr:
.2e} | "
59 f"train loss {record['train_loss']:.4f} | "
60 f"val loss {record['val_loss']:.4f} | "
61 f"intent {record['val_intent_accuracy']:.2f}% | "
62 f"slot F1 {record['val_slot_micro_f1']:.2f}% | "
63 f"combined {selection_score:.2f}% | time {elapsed:.1f
}s"
64 )
65
66 if selection_score > best_score_v2:
67 best_score_v2 = selection_score
68 best_epoch_v2 = epoch
69 epochs_without_improvement = 0
70 torch.save(
71 {
72 "model_state_dict": model_v2.state_dict(),
73 "epoch": epoch,
74 "seed": SEED_V2,
75 "selection_score": best_score_v2,
76 "dev_intent_accuracy": dev_result["intent_acc
uracy"],
77 "dev_slot_micro_f1": dev_result["slot_micro_f
1"],
78 "vocab": vocab,
79 "intent_to_id": intent_to_id,
80 "tag_to_id": tag_to_id,
81 "config": {
82 "rare_threshold": RARE_THRESHOLD,
83 "unk_replace_prob": UNK_REPLACE_PROB,
84 "emb_dropout": EMB_DROPOUT,
85 "attn_dropout": ATTN_DROPOUT,
86 "hidden_dropout": HIDDEN_DROPOUT,
87 "lr": LR_V2,
88 "weight_decay": WEIGHT_DECAY_V2,
89 "warmup_epochs": WARMUP_EPOCHS_V2,
90 "grad_clip": GRAD_CLIP_V2,
91 "max_epochs": NUM_EPOCHS_V2,
92 },
93 },
94 CHECKPOINT_PATH_V2
95 )
96 else:
97 epochs_without_improvement += 1
98 if epochs_without_improvement >= PATIENCE_V2:
99 print(f"Early stopping: no DEV improvement for {P
ATIENCE_V2} epochs.")
100 break
101
102 print(f"Best epoch: {best_epoch_v2}")
103 print(f"Best combined DEV score: {best_score_v2:.2f}%")
104 print(f"Saved checkpoint: {CHECKPOINT_PATH_V2}")

● Edit Notebook(main.ipynb@4ccff27e)
⎿  Updated cell 53526abf:
1 ## D5. Learning Curves: Baseline vs Improvement 1 ##
2
3 Dashed lines are the baseline (`history` from C4). Solid lines
are Improvement 1. If the regularization works, the gap between
train and dev loss should narrow and dev loss should stop clim
bing.
4
5 Note: with dropout and `<UNK>` replacement active, train loss i
s measured on noisier inputs, so it sits higher than the baseli
ne's train loss. That is expected.

● Adding the last cells: comparison plots and the final DEV/TEST evaluation.

● Edit Notebook(main.ipynb@53526abf)
⎿  Updated cell 4414502a:
1 has_baseline_history = "history" in globals() and len(history) > 0
2 epochs_v2 = [x["epoch"] for x in history_v2]
3
4 plt.figure(figsize=(8, 4))
5 if has_baseline_history:
6 epochs_v1 = [x["epoch"] for x in history]
7 plt.plot(epochs_v1, [x["train_loss"] for x in history], "-
-", color="tab:blue", alpha=0.5, label="Baseline train")
8 plt.plot(epochs_v1, [x["val_loss"] for x in history], "--"
, color="tab:orange", alpha=0.5, label="Baseline dev")
9 plt.plot(epochs_v2, [x["train_loss"] for x in history_v2], col
or="tab:blue", label="Improved train")
10 plt.plot(epochs_v2, [x["val_loss"] for x in history_v2], color
="tab:orange", label="Improved dev")
11 plt.xlabel("Epoch")
12 plt.ylabel("Model loss")
13 plt.title("Training and validation loss: baseline vs improveme
nt 1")
14 plt.legend()
15 plt.grid(alpha=0.2)
16 plt.show()
17
18 plt.figure(figsize=(8, 4))
19 if has_baseline_history:
20 plt.plot(epochs_v1, [x["val_intent_accuracy"] for x in his
tory], "--", color="tab:green", alpha=0.5, label="Baseline int
ent acc")
21 plt.plot(epochs_v1, [x["val_slot_micro_f1"] for x in histo
ry], "--", color="tab:red", alpha=0.5, label="Baseline slot F1
")
22 plt.plot(epochs_v2, [x["val_intent_accuracy"] for x in history
\_v2], color="tab:green", label="Improved intent acc")
23 plt.plot(epochs_v2, [x["val_slot_micro_f1"] for x in history_v
2], color="tab:red", label="Improved slot F1")
24 plt.xlabel("Epoch")
25 plt.ylabel("Validation score (%)")
26 plt.title("MASSIVE DEV metrics: baseline vs improvement 1")
27 plt.legend()
28 plt.grid(alpha=0.2)
29 plt.show()

● Edit Notebook(main.ipynb@4414502a)
⎿  Updated cell 88290735:
1 ## D6. Reload Best Checkpoint, Final Evaluation and Comparison ##
2
3 The best DEV checkpoint is reloaded and TEST is evaluated **onc
e**. The baseline row uses `final_dev` / `final_test` from C6 i
f they are in memory. Otherwise it falls back to the recorded b
aseline numbers (DEV 83.23 / 64.21, TEST 82.25 / 58.19).

● Edit Notebook(main.ipynb@88290735) ⎿  Updated cell 2ed42956:
1 checkpoint_v2 = torch.load(CHECKPOINT_PATH_V2, weights_only=Tr
ue)
2 model_v2.load_state_dict(checkpoint_v2["model_state_dict"])
3 model_v2.eval()
4
5 final_dev_v2 = evaluate(model_v2, dev_loader)
6 final_test_v2 = evaluate(model_v2, test_loader)
7 reloaded_score_v2 = (final_dev_v2["intent_accuracy"] + final_d
ev_v2["slot_micro_f1"]) / 2.0
8 assert abs(reloaded_score_v2 - best_score_v2) < 1e-5, "Reloade
d model score does not match best DEV score."
9
10 # Baseline reference numbers
11 if "final_dev" in globals() and "final_test" in globals():
12 base = {
13 "dev_intent": final_dev["intent_accuracy"], "dev_slot"
: final_dev["slot_micro_f1"],
14 "test_intent": final_test["intent_accuracy"], "test_sl
ot": final_test["slot_micro_f1"],
15 }
16 else:
17 base = {"dev_intent": 83.23, "dev_slot": 64.21, "test_inte
nt": 82.25, "test_slot": 58.19}
18
19 impr = {
20 "dev_intent": final_dev_v2["intent_accuracy"], "dev_slot":
final_dev_v2["slot_micro_f1"],
21 "test_intent": final_test_v2["intent_accuracy"], "test_slo
t": final_test_v2["slot_micro_f1"],
22 }
23
24 print(f"Best checkpoint epoch: {checkpoint_v2['epoch']} | Seed
: {checkpoint_v2['seed']}")
25 print(f"Total parameters: {total_params_v2:,} | Trainable: {tr
ainable_params_v2:,}\n")
26 print(f"{'Model':<28}{'DEV intent':>12}{'DEV slot F1':>13}{'TE
ST intent':>13}{'TEST slot F1':>14}")
27 print("-" \* 80)
28 print(f"{'Baseline (C)':<28}{base['dev_intent']:>11.2f}%{base[
'dev_slot']:>12.2f}%"
29 f"{base['test_intent']:>12.2f}%{base['test_slot']:>13.2f
}%")
30 print(f"{'Improvement 1 (D)':<28}{impr['dev_intent']:>11.2f}%{
impr['dev_slot']:>12.2f}%"
31 f"{impr['test_intent']:>12.2f}%{impr['test_slot']:>13.2f
}%")
32 print(f"{'Delta':<28}{impr['dev_intent'] - base['dev_intent']: >+11.2f} {impr['dev_slot'] - base['dev_slot']:>+11.2f} "
33 f"{impr['test_intent'] - base['test_intent']:>+11.2f} {i
mpr['test_slot'] - base['test_slot']:>+12.2f}")
34
35 print("\nSample TEST predictions:\n")
36 show_predictions(model_v2, test_loader, num_examples=5)

● I've added both improvements as a new Part D at the end of main.ipynb. Parts
A–C and the dataset and splits are untouched. I haven't run any of it: there's
no embeddings/ folder or virtual environment in the repo here, so you'll need
to run it on your side.

What's in Part D

Cell: D1
What it does: Rare words → <UNK>. Words seen at most RARE_THRESHOLD = 1 times
in train are swapped for <UNK> with probability 0.5 each time they appear in
a batch. This only happens in a new training collate function,
collate_fn_word_dropout; DEV and TEST loaders are unchanged. It prints how
many words and tokens this affects.
────────────────────────────────────────
Cell: D2
What it does: New model class RegularizedSingleHeadEncoder. Same architecture
as the baseline (1 layer, 1 head, GloVe 100d), plus dropout on the embeddings
(0.2), attention weights (0.1), inside the FFN and on both residual branches
(0.2), and before both classifier heads (0.2). Parameter count stays at
624,071.
────────────────────────────────────────
Cell: D3
What it does: Training setup. AdamW with weight decay 0.01 on the Linear
weight
matrices only (embeddings, biases and LayerNorm are left out, so
rarely-updated GloVe vectors aren't shrunk). One epoch of warmup, then cosine
decay. Gradient clipping at 1.0. Seed reset to 42. Still asserts the 20M
limit.
────────────────────────────────────────
Cell: D4
What it does: Training loop. Up to 40 epochs, stopping early if the DEV score
doesn't improve for 8 epochs. Uses the same selection score as the baseline
and saves to model_checkpoint_v2.pt.
────────────────────────────────────────
Cell: D5
What it does: Loss and metric curves, with the baseline as dashed lines and
the
improved model as solid lines.
────────────────────────────────────────
Cell: D6
What it does: Reloads the best checkpoint, evaluates TEST once, and prints a
baseline vs improved table with the differences.

Everything new has a \_v2 suffix (model_v2, optimizer_v2, history_v2), so none
of the baseline's results in memory are overwritten.

Running it

- Run Parts A–C first. D5 and D6 use the baseline's history, final_dev and
  final_test; if those aren't in memory, D6 falls back to the recorded
  baseline numbers.
- max_length is never defined in the notebook. As the README says, define
  max_length = 64 before A10 or the dataset cells will fail.
- To use the "seen twice or fewer" version, set RARE_THRESHOLD = 2 in D1.

What to check in the results

- Train loss will sit higher than the baseline's, because the inputs are
  deliberately noisier. That's expected.
- The key sign it's working is that DEV loss no longer climbs after epoch 4
  the way the baseline's did.
- This is a single seed, so a difference of 1 point or less could be noise.

Send me the D4 log and the D6 table when it's done, and I'll tune from there
before moving on to the reserved change 3 (making use of GloVe vectors for
words not in the training vocabulary).

✻ Cogitated for 2m 30s · done 8:47 PM

※ recap: The goal is to lift the baseline's 82.25% intent accuracy and 58.19% slot F1, and Part D in main.ipynb now adds rare-word UNK replacement plus dropout and AdamW. Next, run Parts A to D and send me the D4 training log and D6 comparison table. (disable recaps in /config)

❯ ok it improved. what is the other recommendation again, for the improving  
 GloVe part

● Recommendation 3 is about getting more out of GloVe. Right now it has two
problems.

Problem A: test words GloVe already knows get thrown away. The vocabulary
comes from train only, so a word like "tulsa" in TEST that never appeared in
train becomes <UNK>, even if GloVe has a good vector for it. Slot words
(names, places, songs, artists) are the ones hit hardest.

Problem B: fine-tuning moves the vectors. The embeddings are trainable, so
training words drift away from their original GloVe positions. If you simply
added the extra GloVe words back, they would still be in the original space
while the training words had moved, and the model would read them wrongly.

The fix has two parts that go together:

1. Expand the lookup table from GloVe's own word list, not just train words.
   For example, take the top N most frequent GloVe words (GloVe 6B has 400k in
   total). Using GloVe's word list never touches dev/test data; it's the
   external pretrained resource. Mappings, splits and labels stay exactly as
   they are.
2. Freeze the GloVe vectors so every word, seen in train or not, stays in the
   same space. To keep some flexibility, choose one of:
   - Frozen GloVe + a trainable projection: a Linear(100, 100) on top, about
     10k params. Simplest.
   - Frozen GloVe + a small trainable adjustment only for training words,
     initialised to zero. More flexible, slightly more complex.
   - Train-word vectors trainable, extra words frozen, plus a projection. This
     brings back some of the drift problem, so it's the least clean.

Parameter budget (20M limit, frozen vectors count too):

┌───────────────────┬──────────────────┬───────────────────────────┐
│ Extra GloVe words │ Embedding params │ Total model (approx.) │
├───────────────────┼──────────────────┼───────────────────────────┤
│ 50k │ 5.0M │ ~5.1M │
├───────────────────┼──────────────────┼───────────────────────────┤
│ 100k │ 10.0M │ ~10.1M │
├───────────────────┼──────────────────┼───────────────────────────┤
│ 150k │ 15.0M │ ~15.1M │
├───────────────────┼──────────────────┼───────────────────────────┤
│ 190k │ 19.0M │ ~19.1M (cutting it close) │
└───────────────────┴──────────────────┴───────────────────────────┘

I'd go with 100k words. It covers almost all real English words while leaving
room for later changes like the CRF or character CNN.

How it fits with what you just did: the rare-word <UNK> replacement still
helps, because some test words won't be in GloVe at all (typos, unusual names)
and still become <UNK>. Dropout and AdamW carry over unchanged.

Two things to check first:

- That expanding the vocabulary from GloVe is allowed in your course. It's
  standard practice, but some assignments limit you to train words only.
- That you still have embeddings/glove.6B.100d.txt. The full file is needed to
  pick the top 100k words, not just the glove_matrix.npy built from the train
  vocabulary.

If both are fine, I'll add it as Part E on top of Part D, so baseline →
improvement 1 → improvement 2 are all on record. Which freezing option do you
want, or should I default to frozen GloVe + projection?

✻ Worked for 12s · done 9:40 PM

❯ Dont do suggestion 1, it loads GloVe list of words instead of the training  
 input, dont do that.

but can we try the second change, the frozen GloVe with linear projection on  
 top. i recorded initial results already so just modify the part D code  
 directly and I will re-run and compare myself  
 ⎿  ⧉ Selected 1 lines from main.ipynb in Visual Studio Code

● Edit Notebook(main.ipynb@365bf1bb)
⎿  Updated cell 365bf1bb:
1 # Part D. Improvement: Rare-word UNK replacement + Regularizat
ion + Frozen GloVe with projection #
2
3 The baseline (Parts A–C) is left untouched so its results rema
in on record. This part only adds new cells.
4
5 ### Weak points addressed ###
6 1. **`<UNK>` was never trained.** The vocabulary is built from
the train split, so during training no token is ever `<UNK>`.
Its embedding stays random, yet every unseen dev/test word (n
ames, songs, places, the words slots are made of) maps to it.
7 - **Fix (word dropout):** during training only, each token
whose train frequency is `<= RARE_THRESHOLD` is replaced by `<
           UNK>` with probability `UNK_REPLACE_PROB`. The model learns a
useful `<UNK>` vector and learns to tag slots from context.
8 2. **Heavy overfitting.** In the baseline, train loss reaches
0.03 while dev loss rises from 1.05 (epoch 4) to 2.05, with no
regularization anywhere.
9 - **Fix:** dropout on embeddings, attention weights, the FF
N and residual branches, and before both classifier heads. **A
damW** with decoupled weight decay (not applied to embeddings,
biases or LayerNorm), linear warmup + cosine LR decay, gradie
nt clipping and early stopping on DEV.
10 3. **GloVe vectors drift and get memorized.** In the baseline,
about 530k of 624k parameters are the trainable embedding tab
le, so the model memorizes training words.
11 - **Fix:** the GloVe vectors are **frozen**, and a shared t
rainable `Linear(100, 100)` projection, initialised to identit
y, adapts the whole GloVe space to the task at once. The vocab
ulary is still built from the train split only. Rows with no G
loVe vector (`<UNK>` and train words missing from GloVe) stay
trainable through a small separate table, so `<UNK>` can still
be learned via the word dropout in point 1.
12
13 The encoder structure is unchanged: 1 encoder layer, 1 self-at
tention head, GloVe 100d. Training uses the train split only,
tuning/selection uses DEV only, and TEST is evaluated once at
the end.

● Edit Notebook(main.ipynb@0d87cd50)
⎿  Updated cell 0d87cd50:
1 ## D2. Regularized Encoder-only Model with Frozen GloVe + Proj
ection ##
2
3 Same pipeline as B1 (embeddings + learned positions → 1 single
-head self-attention → Add & Norm → FFN → Add & Norm → intent
/ slot heads), with two changes.
4
5 **Embedding layer**
6 `         7 word vector = frozen GloVe row             (words that have a 
           GloVe vector)
         8             | trainable row                (<UNK> and train wo
           rds with no GloVe vector)
         9 X = Dropout( Linear_proj(word vector) + position embedding )
        10`
11 - `word_embeddings`: the full GloVe matrix from A6, **frozen**
. Rows without a GloVe vector are zeroed here.
12 - `oov_embeddings`: a small **trainable** table, only for `<UN
           K>` and train words GloVe doesn't cover, initialised from the
same random vectors A6 used.
13 - `embedding_projection`: a trainable `Linear(100, 100)`, init
ialised to the identity so training starts from the original G
loVe space.
14
15 **Dropout**
16
17 | Location | Rate |
18 |---|---|
19 | Projected embeddings + positions | `EMB_DROPOUT` |
20 | Attention weights (after softmax) | `ATTN_DROPOUT` |
21 | Inside FFN (after ReLU) | `HIDDEN_DROPOUT` |
22 | Attention and FFN outputs before each residual Add | `HIDDEN
           _DROPOUT` |
23 | Before intent and slot classifiers | `HIDDEN_DROPOUT` |
24
25 Dropout is disabled automatically in `model.eval()`, which `ev
           aluate()` already calls.

● Edit Notebook(main.ipynb@ae0c182e)
⎿  Updated cell ae0c182e:
1 class RegularizedSingleHeadEncoder(nn.Module):
2 def **init**(self, glove*vectors, pad_id, max_positions, n
um_intents, num_bio_tags,
3 trainable_word_ids, ffn_dim=200, emb_dropout=
0.2, attn_dropout=0.1, hidden_dropout=0.2):
4 super().**init**()
5 vectors = torch.as_tensor(glove_vectors, dtype=torch.f
loat32).clone()
6 vocab_size, d_model = vectors.shape
7 self.d_model = d_model
8 self.max_positions = max_positions
9
10 # Rows that stay trainable: <UNK> and train words with
no GloVe vector (never <PAD>)
11 trainable_word_ids = sorted(set(int(i) for i in traina
ble_word_ids) - {pad_id})
12 trainable_idx = torch.tensor(trainable_word_ids, dtype
=torch.long)
13
14 # Frozen GloVe table; trainable rows are zeroed here a
nd supplied by oov_embeddings instead
15 frozen_vectors = vectors.clone()
16 frozen_vectors[trainable_idx] = 0.0
17 self.word_embeddings = nn.Embedding.from_pretrained(fr
ozen_vectors, freeze=True, padding_idx=pad_id)
18
19 # Small trainable table: compact row 0 = "not trainabl
e" (always zero), rows 1..N = trainable words
20 oov_index = torch.zeros(vocab_size, dtype=torch.long)
21 oov_index[trainable_idx] = torch.arange(1, len(trainab
le_word_ids) + 1)
22 self.register_buffer("oov_index", oov_index, persisten
t=False)
23 oov_init = torch.zeros(len(trainable_word_ids) + 1, d*
model)
24 oov*init[1:] = vectors[trainable_idx] # same seeded r
andom init as A6
25 self.oov_embeddings = nn.Embedding.from_pretrained(oov
\_init, freeze=False, padding_idx=0)
26
27 # Trainable projection on top of the frozen GloVe spac
e, initialised to identity
28 self.embedding_projection = nn.Linear(d_model, d_model
)
29 nn.init.eye*(self.embedding*projection.weight)
30 nn.init.zeros*(self.embedding*projection.bias)
31
32 # Learned positional embeddings
33 self.position_embeddings = nn.Embedding(max_positions,
d_model)
34 nn.init.normal*(self.position_embeddings.weight, mean=
0.0, std=0.02)
35
36 # Single-head attention parameters
37 self.W_q = nn.Linear(d_model, d_model)
38 self.W_k = nn.Linear(d_model, d_model)
39 self.W_v = nn.Linear(d_model, d_model)
40
41 # Dropout layers (no parameters)
42 self.emb_dropout = nn.Dropout(emb_dropout)
43 self.attn_dropout = nn.Dropout(attn_dropout)
44 self.hidden_dropout = nn.Dropout(hidden_dropout)
45
46 # Add & Norm + FFN + Add & Norm
47 self.norm1 = nn.LayerNorm(d_model)
48 self.ffn = nn.Sequential(
49 nn.Linear(d_model, ffn_dim),
50 nn.ReLU(),
51 nn.Dropout(hidden_dropout),
52 nn.Linear(ffn_dim, d_model)
53 )
54 self.norm2 = nn.LayerNorm(d_model)
55
56 # Two heads sharing the encoder output
57 self.intent_classifier = nn.Linear(d_model, num_intent
s)
58 self.slot_classifier = nn.Linear(d_model, num_bio_tags
)
59
60 def forward(self, input_ids, attention_mask):
61 batch_size, seq_len = input_ids.shape
62 if seq_len > self.max_positions:
63 raise ValueError(f"Sequence length {seq_len} excee
ds maximum {self.max_positions}")
64 attention_mask = attention_mask.bool()
65 position = torch.arange(seq_len, device=input_ids.devi
ce).unsqueeze(0).expand(batch_size, -1)
66
67 # Frozen GloVe row (or trainable row for <UNK>/non-Glo
Ve words) -> projection -> + positions -> dropout
68 word_vectors = self.word_embeddings(input_ids) + self.
oov_embeddings(self.oov_index[input_ids])
69 X = self.emb_dropout(self.embedding_projection(word_ve
ctors) + self.position_embeddings(position))
70
71 # Single-head scaled dot-product attention with paddin
g mask
72 Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
73 scores = torch.matmul(Q, K.transpose(-2, -1)) / math.s
qrt(self.d_model)
74 scores = scores.masked_fill(~attention_mask.unsqueeze(
1), float('-inf'))
75 weights = self.attn_dropout(torch.softmax(scores, dim=
-1))
76 attended = torch.matmul(weights, V)
77
78 # Add & Norm -> FFN -> Add & Norm (dropout on each res
idual branch)
79 H1 = self.norm1(X + self.hidden_dropout(attended))
80 H = self.norm2(H1 + self.hidden_dropout(self.ffn(H1)))
81
82 # Intent: masked mean pooling
83 valid = attention_mask.unsqueeze(-1).to(dtype=H.dtype)
84 pooled = (H \* valid).sum(dim=1) / valid.sum(dim=1).cla
mp(min=1.0)
85 intent_logits = self.intent_classifier(self.hidden_dro
pout(pooled))
86
87 # Slot: per-token classification
88 slot_logits = self.slot_classifier(self.hidden_dropout
(H))
89 return intent_logits, slot_logits

● Edit Notebook(main.ipynb@e96ed8c5)
⎿  Updated cell e96ed8c5:
1 ## D3. Training Setup (AdamW + warmup/cosine schedule) ##
2
3 ### Configurations: ###
4 - Seed: 42 (reset so the comparison with the baseline is fair)
5 - Max epochs: 40, early stopping with patience 8 on the DEV se
lection score
6 - Learning rate: 1e-3, 1 epoch linear warmup, then cosine deca
y to 0
7 - Batch size: 32
8 - Embeddings: GloVe rows frozen. `<UNK>` and train words missi
ng from GloVe (from A6's `found_words`) stay trainable. Shared
`Linear(100, 100)` projection.
9 - AdamW weight decay: 0.01 on Linear weight matrices (includin
g the projection). Embeddings, biases and LayerNorm are exclud
ed.
10 - Gradient clipping: max norm 1.0
11 - Dropout: embeddings 0.2, attention 0.1, hidden/FFN/heads 0.2
12
13 The baseline objects (`model`, `optimizer`, `train_loader`, `h
           istory`) are not overwritten. Everything here uses a `_v2` suf
fix. The frozen GloVe table still counts towards the 20M param
eter limit.

● Edit Notebook(main.ipynb@e3692a04)
⎿  Updated cell e3692a04:
1 random.seed(SEED*V2)
2 np.random.seed(SEED_V2)
3 torch.manual_seed(SEED_V2)
4
5 NUM_EPOCHS_V2 = 40
6 PATIENCE_V2 = 8
7 LR_V2 = 1e-3
8 WEIGHT_DECAY_V2 = 0.01
9 WARMUP_EPOCHS_V2 = 1
10 GRAD_CLIP_V2 = 1.0
11 EMB_DROPOUT = 0.2
12 ATTN_DROPOUT = 0.1
13 HIDDEN_DROPOUT = 0.2
14 CHECKPOINT_PATH_V2 = "model_checkpoint_v2.pt"
15
16 shuffle_generator_v2 = torch.Generator()
17 shuffle_generator_v2.manual_seed(SEED_V2)
18 unk_generator.manual_seed(SEED_V2) # reset after the D1 sanit
y check
19
20 # Training loader uses rare-word <UNK> replacement; DEV/TEST l
oaders are unchanged
21 train_loader_v2 = DataLoader(
22 train_dataset,
23 batch_size=BATCH_SIZE,
24 shuffle=True,
25 collate_fn=collate_fn_word_dropout,
26 num_workers=0,
27 generator=shuffle_generator_v2
28 )
29
30 # Trainable embedding rows: <UNK> + train-vocab words that hav
e no GloVe vector (found_words from A6)
31 trainable_word_ids = [vocab["<UNK>"]] + [
32 idx for word, idx in vocab.items()
33 if word not in ("<PAD>", "<UNK>") and word not in found_wo
rds
34 ]
35 print(f"Frozen GloVe rows: {len(vocab) - len(trainable_word_id
s) - 1:,} | "
36 f"trainable rows (<UNK> + non-GloVe words): {len(trainab
le_word_ids):,}")
37
38 model_v2 = RegularizedSingleHeadEncoder(
39 glove_vectors=embedding_matrix,
40 pad_id=vocab["<PAD>"],
41 max_positions=max_length,
42 num_intents=len(intent_to_id),
43 num_bio_tags=len(tag_to_id),
44 trainable_word_ids=trainable_word_ids,
45 ffn_dim=200,
46 emb_dropout=EMB_DROPOUT,
47 attn_dropout=ATTN_DROPOUT,
48 hidden_dropout=HIDDEN_DROPOUT,
49 )
50
51 total_params_v2 = sum(p.numel() for p in model_v2.parameters()
)
52 trainable_params_v2 = sum(p.numel() for p in model_v2.paramete
rs() if p.requires_grad)
53 frozen_params_v2 = total_params_v2 - trainable_params_v2
54 assert total_params_v2 <= 20_000_000
55
56 # AdamW parameter groups: decay only Linear weight matrices (f
rozen GloVe has no grad and is skipped)
57 decay, no_decay = [], []
58 for name, param in model_v2.named_parameters():
59 if not param.requires_grad:
60 continue
61 if param.ndim >= 2 and "embeddings" not in name:
62 decay.append(param) # attention, projection, FFN,
classifier weights
63 else:
64 no_decay.append(param) # oov/position embeddings, bi
ases, LayerNorm
65
66 optimizer_v2 = torch.optim.AdamW(
67 [
68 {"params": decay, "weight_decay": WEIGHT_DECAY_V2},
69 {"params": no_decay, "weight_decay": 0.0},
70 ],
71 lr=LR_V2,
72 )
73
74 # Linear warmup then cosine decay, stepped once per batch
75 steps_per_epoch = len(train_loader_v2)
76 total_steps = NUM_EPOCHS_V2 * steps*per_epoch
77 warmup_steps = WARMUP_EPOCHS_V2 * steps*per_epoch
78
79 def lr_lambda(step):
80 if step < warmup_steps:
81 return (step + 1) / warmup_steps
82 progress = (step - warmup_steps) / max(1, total_steps - wa
rmup_steps)
83 return 0.5 * (1.0 + math.cos(math.pi \_ min(progress, 1.0))
)
84
85 scheduler_v2 = torch.optim.lr_scheduler.LambdaLR(optimizer_v2,
lr_lambda)
86
87 print("Seed:", SEED_V2)
88 print(f"Max epochs: {NUM_EPOCHS_V2} (patience {PATIENCE_V2}),
batch size: {BATCH_SIZE}, peak LR: {LR_V2}")
89 print(f"AdamW weight decay: {WEIGHT_DECAY_V2} on {sum(p.numel(
) for p in decay):,} params, "
90 f"none on {sum(p.numel() for p in no_decay):,} params")
91 print(f"Dropout: emb {EMB_DROPOUT}, attn {ATTN_DROPOUT}, hidde
n {HIDDEN_DROPOUT}")
92 print(f"Rare-word UNK: count <= {RARE_THRESHOLD}, p = {UNK_REP
LACE_PROB}")
93 print(f"Total parameters: {total_params_v2:,} | Trainable: {tr
ainable_params_v2:,} | Frozen (GloVe): {frozen_params_v2:,}")

● Last edit: recording the embedding settings in the saved checkpoint config.

● Edit Notebook(main.ipynb@4ccff27e)
⎿  Updated cell 4ccff27e:
1 history*v2 = []
2 best_score_v2 = float('-inf')
3 best_epoch_v2 = None
4 epochs_without_improvement = 0
5
6 for epoch in range(1, NUM_EPOCHS_V2 + 1):
7 start = time.perf_counter()
8 model_v2.train() # enables dropout (evaluate() switches
back to eval mode)
9
10 total_train = intent_train = slot_train = 0.0
11 train_examples = 0
12
13 for batch in train_loader_v2:
14 input_ids = batch["input_ids"]
15 attention_mask = batch["attention_mask"]
16 intent_targets = batch["intent_ids"]
17 slot_targets = batch["slot_ids"]
18
19 optimizer_v2.zero_grad(set_to_none=True)
20 intent_logits, slot_logits = model_v2(input_ids, atte
ntion_mask)
21 loss, intent_loss, slot_loss = compute_loss(
22 intent_logits, slot_logits, intent_targets, slot*
targets
23 )
24 loss.backward()
25 torch.nn.utils.clip*grad_norm*(model*v2.parameters(),
GRAD_CLIP_V2)
26 optimizer_v2.step()
27 scheduler_v2.step()
28
29 n = input_ids.size(0)
30 train_examples += n
31 total_train += loss.item() * n
32 intent*train += intent_loss.item() * n
33 slot_train += slot_loss.item() \* n
34
35 # DEV only: no parameter updates
36 dev_result = evaluate(model_v2, dev_loader)
37 selection_score = (dev_result["intent_accuracy"] + dev_re
sult["slot_micro_f1"]) / 2.0
38 elapsed = time.perf_counter() - start
39 current_lr = scheduler_v2.get_last_lr()[0]
40
41 record = {
42 "epoch": epoch,
43 "lr": current_lr,
44 "train_loss": total_train / train_examples,
45 "train_intent_loss": intent_train / train_examples,
46 "train_slot_loss": slot_train / train_examples,
47 "val_loss": dev_result["loss"],
48 "val_intent_loss": dev_result["intent_loss"],
49 "val_slot_loss": dev_result["slot_loss"],
50 "val_intent_accuracy": dev_result["intent_accuracy"],
51 "val_slot_micro_f1": dev_result["slot_micro_f1"],
52 "selection_score": selection_score,
53 "elapsed_time_sec": elapsed
54 }
55 history_v2.append(record)
56
57 print(
58 f"Epoch {epoch:02d}/{NUM_EPOCHS_V2} | lr {current_lr:
.2e} | "
59 f"train loss {record['train_loss']:.4f} | "
60 f"val loss {record['val_loss']:.4f} | "
61 f"intent {record['val_intent_accuracy']:.2f}% | "
62 f"slot F1 {record['val_slot_micro_f1']:.2f}% | "
63 f"combined {selection_score:.2f}% | time {elapsed:.1f
}s"
64 )
65
66 if selection_score > best_score_v2:
67 best_score_v2 = selection_score
68 best_epoch_v2 = epoch
69 epochs_without_improvement = 0
70 torch.save(
71 {
72 "model_state_dict": model_v2.state_dict(),
73 "epoch": epoch,
74 "seed": SEED_V2,
75 "selection_score": best_score_v2,
76 "dev_intent_accuracy": dev_result["intent_acc
uracy"],
77 "dev_slot_micro_f1": dev_result["slot_micro_f
1"],
78 "vocab": vocab,
79 "intent_to_id": intent_to_id,
80 "tag_to_id": tag_to_id,
81 "trainable_word_ids": trainable_word_ids,
82 "config": {
83 "rare_threshold": RARE_THRESHOLD,
84 "unk_replace_prob": UNK_REPLACE_PROB,
85 "frozen_glove": True,
86 "embedding_projection": "Linear(100, 100)
, identity init",
87 "emb_dropout": EMB_DROPOUT,
88 "attn_dropout": ATTN_DROPOUT,
89 "hidden_dropout": HIDDEN_DROPOUT,
90 "lr": LR_V2,
91 "weight_decay": WEIGHT_DECAY_V2,
92 "warmup_epochs": WARMUP_EPOCHS_V2,
93 "grad_clip": GRAD_CLIP_V2,
94 "max_epochs": NUM_EPOCHS_V2,
95 },
96 },
97 CHECKPOINT_PATH_V2
98 )
99 else:
100 epochs_without_improvement += 1
101 if epochs_without_improvement >= PATIENCE_V2:
102 print(f"Early stopping: no DEV improvement for {P
ATIENCE_V2} epochs.")
103 break
104
105 print(f"Best epoch: {best_epoch_v2}")
106 print(f"Best combined DEV score: {best_score_v2:.2f}%")
107 print(f"Saved checkpoint: {CHECKPOINT_PATH_V2}")

● I've changed Part D to freeze GloVe and add a trainable projection on top. The
vocabulary is still built from the train split only, with no GloVe word list
added. I haven't run it, since I don't have GloVe or a Python environment
here.

What changed in Part D

- D2 model:
  - The GloVe vectors are now frozen.
  - A shared trainable Linear(100, 100) projection sits between the word
    vectors and the position embeddings. It starts as the identity, so
    training begins from the original GloVe space.
- One addition you didn't ask for: <UNK> and the roughly 380 training words
  that have no GloVe vector stay trainable, in a small separate table (about
  38k params).
  - Without this, freezing would also lock <UNK> at its random starting
    vector, and the rare-word <UNK> replacement would no longer teach it
    anything.
  - Words that do have a GloVe vector stay fully frozen.
- D3 setup: works out the trainable rows from found_words (from A6) and prints
  total, trainable and frozen parameter counts. The 20M check still counts
  the frozen table. Weight decay now also applies to the projection.
- D4 checkpoint: also saves trainable_word_ids and frozen_glove /
  embedding_projection in its config. D1, D5 and D6 are unchanged.

Things to watch when you compare

- The parameter total rises slightly, to about 672k, but only about 140k are
  now trainable, down from 624k. Expect a smaller gap between train and dev
  loss.
- You may see a small drop in intent accuracy, because the model can no longer
  reshape individual word vectors. If slot F1 or intent falls compared with
  your previous Part D run, the next thing to try is lowering EMB_DROPOUT to
  0.1. With frozen inputs, the earlier dropout level may now be too strong.

✻ Baked for 1m 2s · done 9:53 PM

※ recap: You're improving the intent and slot model; I've updated Part D so GloVe is frozen with a trainable projection on top, keeping a train-only vocabulary. Next, re-run Part D and compare it with your recorded Part D results. (disable recaps in /config)

❯ wait okay i want you to revert the change, it did improve by making the  
 training and validation loss gap smaller, but then its at the cost of  
 training loss going up. it also makes the final end result of intent and F1  
 worse than earlier (without freeze). just revert the part D to not have the  
 freeze

can you put the note that frozen GloVe is tried but it leads to underfit of  
 result (show note of the results, just the smaller gap but the loss  
 sacrifieced and the worse results of intent and F1).

im gonna rerun the model afterwards to get the original unfrozen result

● Reverting the six Part D cells to the unfrozen version, then adding the
experiment note at the end.

● Edit Notebook(main.ipynb@365bf1bb)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@0d87cd50)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@ae0c182e)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@e96ed8c5)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@e3692a04)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@4ccff27e)
⎿  Error editing notebook

● Edit Notebook(main.ipynb@2ed42956) ⎿  Error editing notebook

● None of those edits went through: the notebook changed on disk, probably
because you ran it. Re-reading it first, which may also give me the frozen-run
numbers for the note.

Read 1 file, ran 1 shell command (ctrl+o to expand)

● The run's outputs include the frozen-GloVe numbers. You also added a notes
cell (eb886db0), so I'll read that before editing anything.

Ran 1 shell command (ctrl+o to expand)

● I have what I need. I'll leave your notes cell (eb886db0) untouched and redo
the revert, then add the note with the real frozen-run numbers.

Read 1 file (ctrl+o to expand)

● The Read tool won't load the notebook now that it's grown, so I'll apply the
same changes by editing the notebook JSON with a small Python script.

Made 1 scratchpad edit +263, ran 2 shell commands (ctrl+o to expand)

● I've reverted Part D to the unfrozen version and added the frozen-GloVe note
as a new cell, D7, at the end of Part D.

What changed in the notebook

- D2–D4 are back to the unfrozen version: GloVe is trainable again, the
  projection and the separate <UNK> table are removed, and the extra
  checkpoint keys are gone. I checked that no frozen-GloVe code is left above
  D7.
- D1 (rare-word <UNK> replacement), D5, D6 and your own "Some pointers" cell
  are untouched.
- The Part D intro now points to D7.
- Outputs: the cells still show outputs from the frozen run until you re-run
  them.

What D7 records, using the numbers from your frozen run's output

┌───────────────────────┬──────────┬───────────┬───────────┬────────────┐
│ Model │ DEV │ DEV slot │ TEST │ TEST slot │
│ │ intent │ F1 │ intent │ F1 │
├───────────────────────┼──────────┼───────────┼───────────┼────────────┤
│ Baseline (C) │ 83.33% │ 63.98% │ 82.52% │ 58.75% │
├───────────────────────┼──────────┼───────────┼───────────┼────────────┤
│ Frozen GloVe + │ 84.90% │ 61.97% │ 83.09% │ 58.62% │
│ projection │ │ │ │ │
└───────────────────────┴──────────┴───────────┴───────────┴────────────┘

- Smaller gap: train 0.850 vs DEV 1.082 at the best epoch (33).
- Training loss went up: it was still 0.83 at epoch 40, against 0.03 for the
  baseline.
- Worse results: DEV slot F1 fell below the baseline.
- Conclusion: freezing caused underfitting, so trainable GloVe is kept.

I didn't have your unfrozen Part D numbers, so D7 says "worse than the
unfrozen Part D model" without figures. After you re-run, I can add an
unfrozen row to the D7 table if you send me the D6 output.

The Read tool couldn't load the notebook any more, so I made these edits with
a Python script. A backup of the notebook as it was before is at
C:\Users\micha\AppData\Local\Temp\claude\D--Coding-Encoder-Only-Architect\5f4b
df7a-3873-4216-b8d9-4fa03e5d1030\scratchpad\main.ipynb.bak.

✻ Cogitated for 2m 29s · done 10:19 PM

※ recap: You're improving the model's intent accuracy and slot F1 without touching the baseline, and Part D is back to trainable GloVe with the frozen-GloVe attempt recorded in D7. Next, re-run Part D to get the unfrozen results.

---

## 12 November 2026 — evaluation and reproducibility follow-up

**Tool:** OpenAI Codex. The exact model identifier was not recorded in this entry.

**User requests:**

> In the SC4002-1 folder main.ipynb, help to update validation scores to ensure everything is up to date. Run it through from start to end, and remove any manually entered score fallbacks. Report the final findings in summary

> Can you resolve the multiset slot scorer and update the notebook then?

**Work and reasoning:** The notebook's comparison cell had a manually entered baseline-score fallback, and saved outputs from different runs disagreed. Codex removed the fallback and reran both stages. It then pointed out that the shared slot evaluator counted duplicate type-and-span pairs as a multiset, whereas the assignment specifies a set of pairs for each utterance. Following the user's request, Codex changed the shared evaluator to deduplicate exact pairs within each utterance and pool true positives, false positives and false negatives across utterances. Checks for duplicate gold pairs, duplicate predictions, mismatched types and spans, and empty sets passed.

Codex reran the full notebook with the corrected scorer, selecting both checkpoints using the average of validation intent accuracy and slot micro-F1. Both checkpoint reload checks passed; the notebook contains the actual training logs and outputs. The corrected scoring changed the baseline's selected checkpoint from epoch 12 in the immediately preceding multiset-scored run to epoch 17. The improved system selected epoch 23.

| Model | Best epoch | Validation intent accuracy | Validation slot micro-F1 | Test intent accuracy | Test slot micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 17 | 83.42% | 63.80% | 82.48% | 58.95% |
| Improved | 23 | 86.82% | 66.83% | 85.74% | 64.32% |

The historical frozen-GloVe experiment in notebook section D7 used the earlier multiset scorer; its recorded figures are not directly comparable with the corrected results. The test split had already been inspected in earlier experiments, so it should not be described as previously untouched. This entry is a summary of the collaboration, not a complete verbatim chat export.
