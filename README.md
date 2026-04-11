# MSDS688 Week 5 — Building Conversational Bots with DialoGPT

Regis University — Master of Science in Data Science
**Author**: Morgan Cooper

## Overview

This assignment fine-tunes Microsoft's [DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) to build persona-grounded conversational chatbots. There are two parts:

- **Part 1 — *Follow Me***: a guided walkthrough that fine-tunes DialoGPT on the [DailyDialog](https://huggingface.co/datasets/OpenRL/daily_dialog) dataset and demonstrates a basic multi-turn chat loop.
- **Part 2 — *Your Turn***: an independent implementation that fine-tunes DialoGPT on a persona-chat dataset (`personality.csv`) and produces a chatbot that stays grounded in a randomly-selected persona.

## Repository layout

| File | Description |
|---|---|
| [`week5_Morgan_Cooper.ipynb`](week5_Morgan_Cooper.ipynb) | Full assignment notebook (Part 1 + Part 2) with completed TODOs and executed outputs |
| [`personality.csv`](personality.csv) | Persona-chat dataset used by Part 2 (one persona + multi-turn chat per row) |
| `README.md` | This file |

## Running the notebook

**GPU required.** The notebook is designed to run in Google Colab:

1. Upload `week5_Morgan_Cooper.ipynb` and `personality.csv` to Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the first code cell (the `pip install` cell).
4. **Runtime → Restart session** (required — the install replaces `transformers` in memory).
5. **Runtime → Run all**.

Each part takes about one to two minutes to fine-tune on a T4.

## Key technical decisions

The notebook documents these inline alongside the code, but the main engineering choices worth calling out:

- **Pinned `transformers==4.46.3` / `datasets==2.21.0`**. The latest 5.x `transformers` silently breaks DialoGPT's `lm_head` weight tying and causes fine-tuning to collapse the model into producing garbage output. The pin is mandatory, not stylistic.
- **Label-masked tokenization**. Training sequences are formatted as alternating `User:` and `Bot:` turns (with a `Persona:` prefix in Part 2). Every token that is not part of a `Bot:` turn is masked with `-100` (PyTorch's cross-entropy ignore index), so only the bot's responses contribute to the loss. Without this, ~90% of the gradient signal comes from padding and user turns, and the model collapses.
- **`default_data_collator` instead of `DataCollatorForLanguageModeling`**. The LM collator overwrites the `labels` field with a copy of `input_ids`, which would undo the masking above.
- **Greedy decoding (`do_sample=False`)**. On Colab, `torch.multinomial` can fire a CUDA device-side assert when nucleus sampling filters mask every candidate token. Greedy decoding uses `argmax` and bypasses that failure mode entirely. Replies are deterministic but the cells actually finish running.

## Sample outputs

**Part 1 — DailyDialog fine-tune**

```
You: Hi, how are you?
Bot: I'm good, thanks. How about you?

You: What do you like to do for fun?
Bot: Well, I like playing tennis and playing with friends.

You: Tell me something interesting.
Bot: Sure, I like to play basketball.
```

**Part 2 — Persona-grounded chatbot**

```
Persona: i just started college. i have 3 science classes. i work part time in
the campus library. i am living at home but hope to live in the dorms next year.

You: Hi, how are you?
Bot: i am doing great, just finished my first semester of college.

You: What do you like to do for fun?
Bot: reading, watching tv, playing video games, and going out with friends.

You: Tell me something interesting about yourself.
Bot: nothing too interesting. what kind of school do you want to go to?
```

The Part 2 output is grounded in the given persona — the persona mentions college, and the bot's first reply references finishing a semester of college. The final reply even asks a natural follow-up question.
