---
inference: false
license: other
datasets:
- jondurbin/airoboros-gpt4-1.2
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# John Durbin's Airoboros 7B GPT4 1.2 GPTQ

These files are GPTQ 4bit model files for [John Durbin's Airoboros 7B GPT4 1.2](https://huggingface.co/jondurbin/airoboros-7b-gpt4-1.2).

It is the result of quantising to 4bit using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

## Repositories available

* [4-bit GPTQ models for GPU inference](https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.2-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGML models for CPU+GPU inference](https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.2-GGML)
* [Unquantised fp32 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/jondurbin/airoboros-7b-gpt4-1.2)

## Prompt template

```
A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.
USER: prompt
ASSISTANT:
```

## How to easily download and use this model in text-generation-webui

Please make sure you're using the latest version of text-generation-webui

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/airoboros-7B-gpt4-1.2-GPTQ`.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done"
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `airoboros-7B-gpt4-1.2-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to and should not set manual GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!

## How to use this GPTQ model from Python code

First make sure you have [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) installed:

`pip install auto-gptq`

Then try the following example code:

```python
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

model_name_or_path = "TheBloke/airoboros-7B-gpt4-1.2-GPTQ"
model_basename = "airoboros-7b-gpt4-1.2-GPTQ-4bit-128g.no-act-order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# Note: check the prompt template is correct for this model.
prompt = "Tell me about AI"
prompt_template=f'''USER: {prompt}
ASSISTANT:'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
```

## Provided files

**airoboros-7b-gpt4-1.2-GPTQ-4bit-128g.no-act-order.safetensors**

This will work with AutoGPTQ and CUDA versions of GPTQ-for-LLaMa. There are reports of issues with Triton mode of recent GPTQ-for-LLaMa. If you have issues, please use AutoGPTQ instead.

* `airoboros-7b-gpt4-1.2-GPTQ-4bit-128g.act.order.safetensors`
  * Works with AutoGPTQ in CUDA or Triton modes.
  * Works with GPTQ-for-LLaMa in CUDA mode.  May have issues with GPTQ-for-LLaMa Triton mode.
  * Works with text-generation-webui, including one-click-installers.
  * Parameters: Groupsize = 128. Act Order / desc_act = True.

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute.

Thanks to the [chirper.ai](https://chirper.ai) team!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Sam, theTransient, Jonathan Leane, Steven Wood, webtim, Johann-Peter Hartmann, Geoffrey Montalvo, Gabriel Tamborski, Willem Michiel, John Villwock, Derek Yates, Mesiah Bishop, Eugene Pentland, Pieter, Chadd, Stephen Murray, Daniel P. Andersen, terasurfer, Brandon Frisco, Thomas Belote, Sid, Nathan LeClaire, Magnesian, Alps Aficionado, Stanislav Ovsiannikov, Alex, Joseph William Delisle, Nikolai Manek, Michael Davis, Junyu Yang, K, J, Spencer Kim, Stefan Sabev, Olusegun Samson, transmissions 11, Michael Levine, Cory Kujawski, Rainer Wilmers, zynix, Kalila, Luke @flexchar, Ajan Kanaga, Mandus, vamX, Ai Maven, Mano Prime, Matthew Berman, subjectnull, Vitor Caleffi, Clay Pascal, biorpg, alfie_i, 阿明, Jeffrey Morgan, ya boyyy, Raymond Fosdick, knownsqashed, Olakabola, Leonard Tan, ReadyPlayerEmma, Enrico Ros, Dave, Talal Aujan, Illia Dulskyi, Sean Connelly, senxiiz, Artur Olbinski, Elle, Raven Klaugh, Fen Risland, Deep Realms, Imad Khwaja, Fred von Graf, Will Dee, usrbinkat, SuperWojo, Alexandros Triantafyllidis, Swaroop Kallakuri, Dan Guido, John Detwiler, Pedro Madruga, Iucharbius, Viktor Bowallius, Asp the Wyvern, Edmond Seymore, Trenton Dambrowitz, Space Cruiser, Spiking Neurons AB, Pyrater, LangChain4j, Tony Hughes, Kacper Wikieł, Rishabh Srivastava, David Ziegler, Luke Pendergrass, Andrey, Gabriel Puliatti, Lone Striker, Sebastain Graf, Pierre Kircher, Randy H, NimbleBox.ai, Vadim, danny, Deo Leter


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: John Durbin's Airoboros 7B GPT4 1.2


### Overview

This is a qlora fine-tuned 7b parameter LlaMa model, using completely synthetic training data created gpt4 via https://github.com/jondurbin/airoboros

This is mostly an extension of [1.1](https://huggingface.co/jondurbin/airoboros-7b-gpt4-1.1), but with thousands of new training data and an update to allow "PLAINFORMAT" at the end of coding prompts to just print the code without backticks or explanations/usage/etc.

The dataset used to fine-tune this model is available [here](https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.2), with a specific focus on:
- coding
- math/reasoning (using orca style ELI5 instruction/response pairs)
- trivia
- role playing
- multiple choice and fill-in-the-blank
- context-obedient question answering
- theory of mind
- misc/general

This model was fine-tuned with a fork of [qlora](https://github.com/jondurbin/qlora), which among other things was updated to use a slightly modified vicuna template to be compatible with the previous versions:

```
A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. USER: [prompt] ASSISTANT:
```

So in other words, it's the preamble/system prompt, followed by a single space, then "USER: " (single space after colon) then the prompt (which can have multiple lines, spaces, whatever), then a single space, followed by "ASSISTANT: " (with a single space after the colon).

### Usage

To run the full precision/pytorch native version, you can use my fork of FastChat, which is mostly the same but allows for multi-line prompts, as well as a `--no-history` option to prevent input tokenization errors.
```
pip install git+https://github.com/jondurbin/FastChat
```

Be sure you are pulling the latest branch!

Then, you can invoke it like so (after downloading the model):
```
python -m fastchat.serve.cli \
  --model-path airoboros-7b-gpt4-1.2 \
  --temperature 0.5 \
  --max-new-tokens 2048 \
  --no-history
```

Alternatively, please check out TheBloke's quantized versions:

- https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.2-GPTQ
- https://huggingface.co/TheBloke/airoboros-7B-gpt4-1.2-GGML

### Coding updates from gpt4/1.1:

I added a few hundred instruction/response pairs to the training data with "PLAINFORMAT" as a single, all caps term at the end of the normal instructions, which produce plain text output instead of markdown/backtick code formatting.

It's not guaranteed to work all the time, but mostly it does seem to work as expected.

So for example, instead of:
```
Implement the Snake game in python.
```

You would use:
```
Implement the Snake game in python.  PLAINFORMAT
```

### Other updates from gpt4/1.1:

- Several hundred role-playing data.
- A few thousand ORCA style reasoning/math questions with ELI5 prompts to generate the responses (should not be needed in your prompts to this model however, just ask the question).
- Many more coding examples in various languages, including some that use specific libraries (pandas, numpy, tensorflow, etc.)
