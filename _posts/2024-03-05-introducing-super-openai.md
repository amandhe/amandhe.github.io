---
layout: post
title: "Introducing superopenai"
date: 2024-03-05
categories: superopenai llm python openai
---

_TL;DR: [`superopenai`](https://github.com/villagecomputing/superopenai) is a minimal library for logging and caching LLM requests and responses during development, enabling visibility and rapid iteration._

[**Star us!**](https://github.com/villagecomputing/superopenai)

## Visibility is all you need

If you’ve tried developing products with LLMs your experience has probably been something like this:

<blockquote class="twitter-tweet tw-align-center"><p lang="en" dir="ltr">debugging is. there are just _so many_ instructions i could change, and of course I’m going to try them all because I’m a software engineer and i like to be thorough. this story has no ending, sometimes i feel I’m kidding myself believing that LLMs can be used like software APIs</p>&mdash; Shreya Shankar (@sh_reya) <a href="https://twitter.com/sh_reya/status/1762211784633811369?ref_src=twsrc%5Etfw">February 26, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Building LLM-based software is an iterative process. You build something, try it out, observe cases where it doesn’t work and then try to make improvements to address those cases. The end result of this is often a jumbled mess of print statements and prompt iteration.

This problem gets exacerbated when building multistep pipelines with RAG or agentic workflows. In these flows, prompts are generated dynamically. You don’t even know what the LLM is seeing so you insert more print statements and debugging code.

To ease the complexity burden, people often turn to “llm-magic” libraries like langchain, guardrails, and instructor. These libraries can accelerate writing the first version of your code, but can have adverse side effects. When using them, it’s helpful to have a “[show me the prompt](https://hamel.dev/blog/posts/prompt/)” attitude. That requires more boilerplate, more debug code, and more print statements.

<div style="text-align: center;"><img src="/assets/images/fushowme.jpeg" alt="Show me the prompt!" title="Show me the prompt!" style="width: 60%;" /></div>

<br>

And when you change a single prompt and re-run your code, you have to wait for every single LLM request to be re-executed. Good luck iterating quickly when you have to wait minutes between each iteration. Changed some prompts and your results got worse? 🤞 Cross your fingers that the previous prompts are still saved in your undo history.

Once you build something that works with a powerful model, the next step is to optimize it for cost and speed. Models that are more accurate tend to be more expensive and slower. The only way to evaluate this tradeoff is to measure it. Oh, you only logged `response.choices[0].message.content`? More debug code and print statements.

Cost and latency can be exacerbated by the use of these “llm-magic” libraries, which will often make many LLM requests under the hood, leaving you wondering what’s really going on.

**`superopenai` is a local obvervability tool that logs prompts, responses, latency, cost and token usage. It also caches LLM requests and responses when the request is identical and `temperature=0`. Initialize it once, and everything gets logged to a local file and cached in-memory automatically.**

There's a lot of alpha in just looking at your prompts.

<blockquote class="twitter-tweet tw-align-center"><p lang="en" dir="ltr">a lot of alpha in just looking at your prompts! <a href="https://t.co/HAB4PUF2mM">https://t.co/HAB4PUF2mM</a></p>&mdash; Harrison Chase (@hwchase17) <a href="https://twitter.com/hwchase17/status/1762254441770860938?ref_src=twsrc%5Etfw">February 26, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

**Do we need another LLM observability tool?**

Good question. There are plenty of good observability and tracing tools for monitoring your LLM apps in production - [Braintrust](https://braintrustdata.com/), [Traceloop](https://www.traceloop.com/), [Langsmith](https://docs.smith.langchain.com/), [Helicone](https://www.helicone.ai/), and probably others I'm missing.

`superopenai` is not meant for production apps, but for development. When you fire up a Jupyter notebook to experiment with something quickly, or when you're just starting to build an idea from scratch, it's helpful to look at your logs locally rather than set up a remote observability tool.

If you're doing something that requires 100% privacy and you can't send data to any third parties, `superopenai` may also be helpful to roll your own logging and observability solution.

## Introducing superopenai

_superopenai gives you logging and caching superpowers when developing with the openai sdk_

`superopenai` gives you visibility into what's really going on under the hood. It wraps the `openai` client and logs requests and responses to a convenient `Logger` object. It also logs _cost, token usage, latency_ and computes some useful statistics over them. It can optionally cache repeat identical requests when `temperature=0` to prevent unnecessary waiting when repeatedly testing/iterating on your code in development.

You can install it with `pip install superopenai`. Here's a simple example of how to use it:

```python
from openai import OpenAI
from superopenai import init_logger, init_superopenai

init_superopenai()
client = OpenAI()

with init_logger() as logger:
  client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "user", "content": "What's the capital of France?"}
    ])
  for log in logger.logs:
    print(log)
```

This will output

```
+-----------+----------------------------------------------+
| Messages  | - user: What's the capital of France?        |
+-----------+----------------------------------------------+
| Arguments | - model: gpt-4-1106-preview                  |
+-----------+----------------------------------------------+
| Output    | - assistant: The capital of France is Paris. |
+-----------+----------------------------------------------+
| Metadata  | - Cost: $0.00035                             |
|           | - Prompt tokens: 14                          |
|           | - Completion tokens: 7                       |
|           | - Total tokens: 21                           |
|           | - Start time: 1709914488.7480488             |
|           | - Latency: 0.7773971557617188                |
+-----------+----------------------------------------------+
| Cached    | False                                        |
+-----------+----------------------------------------------+
```

### Aggregate Statistics

When runnning a chain or agent with multiple LLM calls, it's useful to look at summary statistics over all the calls rather than individual ones, using the `summary_statistics` function on the `Logger` object.

```python
with init_logger() as logger:
  client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "user", "content": "What's the capital of France?"}
    ]
  )
  print(logger.summary_statistics())
```

This will output

```
+----------------------------+----------------------------+
|      Number of Calls       |             1              |
|       Number Cached        |             1              |
+----------------------------+----------------------------+
|            Cost            |          $0.00035          |
+----------------------------+----------------------------+
|       Prompt Tokens        |             14             |
|     Completion Tokens      |             7              |
|        Total Tokens        |             21             |
+----------------------------+----------------------------+
|   Prompt Tokens by Model   | {'gpt-4-1106-preview': 14} |
| Completion Tokens by Model | {'gpt-4-1106-preview': 7}  |
|   Total Tokens by Model    | {'gpt-4-1106-preview': 21} |
+----------------------------+----------------------------+
|       Total Latency        |   3.981590270996094e-05    |
|      Average Latency       |   3.981590270996094e-05    |
|  Average Latency (Cached)  |   3.981590270996094e-05    |
| Average Latency (Uncached) |             0              |
+----------------------------+----------------------------+
```

### Caching

Requests are cached in memory using `cachetools` if the request params are exactly the same, the same `openai` client object is used, and `temperature` is set to `0`. The latter condition is because if `temperature` is non-zero then the LLM may return a different result even for the same params.

Caching is enabled by default and the default cache size is 1000 items. It can be disabled and the cache size can be changed when initializing `superopenai`

```python
from superopenai import init_superopenai

init_superopenai(enable_caching=True, cache_size=100)
```

### Other features

**Streaming and async support**.

Logging works in streaming mode (setting `stream=True` in the chat completion `create` request) as well as when using the async chat completion api.

**Nested loggers**

Loggers can be nested arbitrarily deep within other loggers. For example, in an agentic system where one agent instantiates another agent

```python
def other_agent():
  with init_logger() as logger: # make some LLM calls
    print(logger.logs) # will only include LLM calls made by other_agent

def agent():
  with init_logger() as logger: # make some LLM calls
    other_agent()
    print(logger.logs) # will only include LLM calls made by agent, not other_agent
```

**Compatible with langchain, etc.**

`superopenai` is fully compatible with `langchain`, `llama-index`, `instructor`, `guidance`, `DSpy` and most other third party libraries. Just call `init_superopenai` before using any other library.

## Example: Building and testing a RAG pipeline with langchain

Let's walk through the process of building a simple Q&A bot to answer questions over Tesla's 2023 10-K annual filing. We'll start with [this code](https://python.langchain.com/docs/use_cases/question_answering/quickstart) from the Langchain docs and make some minor changes to make it work with PDFs. The final code is available as a [GitHub gist](https://gist.github.com/amandhe/d33fcf44aa6a2a77c00c014bf25405bd).

Now we'll ask some questions. I use the following snippet to log all LLM requests using `superopenai` and print out the cost and latency

```python
with init_logger() as logger:
  print(rag_chain.invoke(question))
  print("Cost: $", logger.summary_statistics().cost)
  print("Latency: ", logger.summary_statistics().total_latency)
```

We'll ask it the following question:

> in what ways does the inflation reduction act (IRA) benefit Tesla? give me specific numbers and facts. return 5 bullet points maximum

And we get this response:

> - Qualifying Tesla customers may receive up to $7,500 in federal tax credits for the purchase of qualified electric vehicles in the U.S. through 2032.
> - The IRA promotes clean energy, electric vehicles, battery, and energy storage manufacture or purchase, which aligns with Tesla's business model.
> - The IRA is expected to materially affect Tesla's consolidated financial statements, potentially reducing material costs in their consolidated statement of operations.
> - Tesla may benefit from incentives provided to manufacturers who assemble domestically and have local suppliers, which could apply to Tesla's operations.
> - The impact of the IRA on Tesla's financials was primarily a reduction of material costs for the year ended December 31, 2023, indicating a financial benefit from the act.

> Cost: $ 0.0187

On first glance this seems pretty good. The response mentioned the $7500 tax credit some Tesla vehicles qualify for. However, it didn't specifically mention that Tesla has a major energy storage business that benefits from the IRA. It also didn't mention the precise amount that energy storage benefits from IRA.

Now let's change `K` to `20`, re-run the code and ask the same question. This time we get:

> - Qualifying Tesla customers may receive up to $7,500 in federal tax credits for the purchase of qualified electric vehicles in the U.S. through 2032.
> - The IRA includes incentives to promote clean energy, electric vehicles, battery, and energy storage manufacture or purchase, which could reduce material costs in Tesla's consolidated statement of operations.
> - The IRA provides benefits to manufacturers who assemble domestically and have local suppliers, which may apply to Tesla's operations.
> - Standalone energy storage technology is eligible for a tax credit under the IRA, which could benefit Tesla's energy storage solutions.
> - Tesla may experience a margin benefit from IRA manufacturing credits earned, although specific numbers regarding this benefit are not provided in the context.

> Cost: $ 0.03224

This time the response did include a bullet point about energy storage! However, it still didn't mention the precise subsidy the IRA provides for energy storage. I know that Tesla's energy storage is eligible for between 6% and 50% of the expenses in tax credits. Can we get the model to include that in its response, without specifically asking for it? Note that the cost almost doubled with twice as many input chunks.

One thing we can do is to try Langchain's `MultiQueryRetriever` which expands the input question and asks it 3 different ways (using an LLM request).

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 20}), llm=llm
)
rag_chain = (
    {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

With this chain and the same question, we get:

> - Qualifying Tesla customers may receive up to $7,500 in federal tax credits for the purchase of qualified electric vehicles in the U.S. through 2032.
> - The IRA includes incentives to promote clean energy, electric vehicles, battery, and energy storage manufacture or purchase, which could reduce material costs in Tesla's consolidated statement of operations.
> - The IRA provides tax credits for standalone energy storage technology, which could benefit Tesla's energy storage products.
> - The IRA includes multiple incentives that are expected to materially affect Tesla's consolidated financial statements, although specific impacts are subject to future guidance releases.
> - Tesla's costs of revenue were positively impacted by the IRA manufacturing credits earned during the current year, contributing to a reduction in material costs and lower manufacturing costs from better fixed cost absorption.
>   (Note: The provided context does not contain specific numbers regarding the exact financial impact of the IRA on Tesla's financial statements beyond the $7,500 tax credit for consumers and the general statement about the positive impact on material costs and manufacturing credits.)

> Cost: $ 0.05087

Still no luck, even though the cost went up another 50%. One last try, let's double the chunk size to `1000`, overlap to `200` and number of chunks to `40`

> - Qualifying Tesla customers may receive up to $7,500 in federal tax credits for the purchase of qualified electric vehicles in the U.S. through 2032.
> - Standalone energy storage technology is eligible for a tax credit between 6% and 50% of qualified expenditures, which may benefit Tesla customers or Tesla directly if they own the systems.
> - Solar energy systems are eligible for a tax credit between 6% and 70% of qualified commercial or residential expenditures, which may benefit Tesla customers or Tesla directly if they own the systems.
> - The IRA incentives are expected to materially affect Tesla's consolidated financial statements, primarily as a reduction of material costs in their consolidated statement of operations.
> - Tesla's business and products are impacted by various government credits, incentives, and policies, which may include benefits to manufacturers who assemble domestically and have local suppliers.

> Cost: $ 0.11791

Aha! This time it worked. Notice that cost more than doubled from the previous attempt, but this is definitely the best answer yet. This is the cost-accuracy tradeoff we discussed above - more on that soon ;)

## Contributing

`superopenai` is free, open-source, and licensed under the MIT license. We welcome contributions from the community.

Some ideas for future directions:

- Port to TS and other languages
- Add retries
- Add detailed function tracing
- Disk and remote caching
- Thread-safe caching
- Integrate with 3rd party hosted logging services

Or, you can always contribute by [giving us a star](https://github.com/villagecomputing/superopenai) :)
