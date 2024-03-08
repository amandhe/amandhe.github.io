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

{% highlight python %}
def other_agent():
with init_logger() as logger: # make some LLM calls
print(logger.logs) # will only include LLM calls made by other_agent

def agent():
with init_logger() as logger: # make some LLM calls
other_agent()
print(logger.logs) # will only include LLM calls made by agent, not other_agent

{% endhighlight %}

**Compatible with langchain, etc.**

`superopenai` is fully compatible with `langchain`, `llama-index`, `instructor`, `guidance`, `DSpy` and most other third party libraries. Just call `init_superopenai` before using any other library.

## Example: Building a RAG pipeline with langchain

Let's walk through the process of building a simple Q&A bot to answer questions over Tesla's 2023 10-K annual filing. We'll use [this code](https://python.langchain.com/docs/use_cases/question_answering/quickstart) from the Langchain docs with some minor changes to work with PDFs rather than websited.

```python
from superopenai import init_superopenai, init_logger
init_superopenai()

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

CHUNKSIZE = 1000
CHUNKOVERLAP = 200
MODEL = "gpt-4-1106-preview"
K = 10

loader = PyPDFLoader("./tesla_10k.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNKSIZE, chunk_overlap=CHUNKOVERLAP)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": K})
prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
llm = ChatOpenAI(model_name="MODEL", temperature=0)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Now, I ask the question `how did regulatory tax credits effect tesla's financials for the year compared to last year?`

```python
rag_chain.invoke("how did regulatory tax credits effect tesla's financials for the year compared to last year?")
# Response: "Regulatory tax credits had a minor positive effect on Tesla's financials in 2023 compared to 2022. Automotive regulatory credits revenue increased by $14 million, or 1%, from $1,776 million in 2022 to $1,790 million in 2023. This increase is relatively small in the context of Tesla's overall revenue growth."
```

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
