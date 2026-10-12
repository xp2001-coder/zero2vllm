# 2.7 实验：从另一个终端逐项读取生成结果

[正文](README.md) · [练习与答案](exercises.md) · [运行记录](results/README.md)

所有命令在仓库根目录运行。本节先用两个小例子理解异步迭代和拼帧，再启动真实 HTTP 服务，最后运行故障与回收案例。默认不需要 GPU、PyTorch 或模型下载。

## 1. 准备环境

Python 3.11 或 3.12。已有课程虚拟环境可直接使用；Windows Git Bash：

```bash
source .venv/Scripts/activate
python -m pip install uvicorn==0.35.0
python -c "import sys, uvicorn; print(sys.executable); print(uvicorn.__version__)"
```

PowerShell 使用：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install uvicorn==0.35.0
```

Linux 激活命令为：

```bash
source .venv/bin/activate
```

首次创建环境可先运行 `python -m venv .venv`。不激活时，Windows 把后文 python 替换为 `.venv/Scripts/python.exe`，Linux 换为 `.venv/bin/python` 即可。若 PowerShell 拒绝执行激活脚本，也可直接用这个完整解释器路径，无需修改系统策略。

只运行下面两个无网络短示例时，无需安装 Uvicorn。可选前缀后端沿用 2.6 的 CPU PyTorch 环境，安装方法见 [2.6 实验](../06-prefix-caching/lab.md)。

## 2. 先不启动 HTTP：看异步迭代与提前退出

```bash
python docs/chapters/part-02-engine/07-http-and-streaming-output/examples/walk_events.py
```

实际输出：

```text
full: accepted seq= 0
full: token seq= 1
full: token seq= 2
full: token seq= 3
full: done seq= 4
full output: [12, 3, 8]
early: accepted
early: token
after early close: open_streams=0 used_tokens=0 status=cancelled
PASS: async iteration and explicit cleanup.
```

按四个观察点读：

1. `accepted` 是事件，不是 token，因此输出列表没有多出一项。
2. 三次 token 之后才出现 `done`；序号 0—4 连续。
3. 第二条请求的输出上限是 20，调用方看到第一项就 `break`。
4. `aclosing` 立即关闭异步生成器，触发 `finally`，取消和 `detach` 后流与额度都归零。

这个示例没有经过网络，只验证结果通道与所有权。它不能替代后面的真实 HTTP 断连实验。

## 3. 一个字节一个字节地读 SSE

```bash
python docs/chapters/part-02-engine/07-http-and-streaming-output/examples/inspect_frames.py
```

实际输出：

```text
complete event: accepted seq= 0
complete event: token seq= 1
complete event: done seq= 2
truncated frame: rejected
PASS: byte fragments preserve event boundaries and UTF-8.
```

示例在 `accepted` 数据里包含中文“开始”，故意每次只把一个字节交给解析器。只有遇到事件末尾空行后才打印 complete event；中文多字节字符也完整恢复。最后故意截断一条 `done` 帧，解析器必须拒绝。

可以打开源码，把每次输入改成 2 或 7 字节，或一次交入全部数据。最终事件应相同，不能依赖某个“刚好完整”的网络分块。

## 4. 启动真实服务：终端 A

```bash
python -m zero2vllm.serving.server --step-delay 0.05
```

等待日志出现应用启动完成，以及本机 8000 端口的监听信息。这个终端持续运行，不会立刻返回命令提示符。

step-delay=0.05 让每轮推进之后人为等待一小段时间，方便肉眼看见流式输出、及时关闭客户端。它不参与模型数值计算，也不是测到的模型延迟。正常不需要演示时可省略。

服务默认只有一个进程，监听 127.0.0.1:8000。若端口占用：

```bash
python -m zero2vllm.serving.server --port 8001 --step-delay 0.05
```

此时客户端也要把地址改为 8001。不要只改一边。

## 5. 发起生成：终端 B

另开终端，进入同一个仓库、激活同一环境，执行：

```bash
python -m zero2vllm.serving.client
```

会逐行打印 JSON 事件。`request_id` 每次不同，不应逐字符复制比较。忽略随机 ID 后，主要字段应为：

| event | `seq` | `index` | `token_id` | 终态 |
| --- | --- | --- | --- | --- |
| `accepted` | 0 | — | — | 尚未结束 |
| token | 1 | 0 | 12 | 尚未结束 |
| token | 2 | 1 | 3 | 尚未结束 |
| token | 3 | 2 | 8 | 尚未结束 |
| `done` | 4 | — | — | `finished` / length / `output_tokens`=3 |

客户端默认提交 [1,2,3,4,5,6]，最多生成 3 项。修改输入：

```bash
python -m zero2vllm.serving.client --prompt 1 2 3 4 9 10 --max-new-tokens 3
```

预期生成 [14,13,1]。

如果服务端使用 8001：

```bash
python -m zero2vllm.serving.client --url http://127.0.0.1:8001/generate
```

### 用 curl 查看原始 SSE 文本

Git Bash / Linux：

```bash
curl -N http://127.0.0.1:8000/generate -H "Content-Type: application/json" --data-binary "@docs/chapters/part-02-engine/07-http-and-streaming-output/data/request.json"
```

Windows PowerShell 使用 `curl.exe`，避免旧环境把 curl 解析成其他命令的别名：

```powershell
curl.exe -N http://127.0.0.1:8000/generate -H "Content-Type: application/json" --data-binary "@docs/chapters/part-02-engine/07-http-and-streaming-output/data/request.json"
```

数据来自 [request.json](data/request.json)。这里每条事件会分成 event 行、data 行和空行；Python 客户端把它解析成一个对象后才打印，因此显示形式不同。

## 6. 客户端提前退出，检查服务是否继续可用

保持服务端运行：

```bash
python -m zero2vllm.serving.client --max-new-tokens 20 --stop-after 1
```

客户端读到第一条 token 事件后关闭连接，打印：

```text
Client closed early; no successful completion claimed.
```

这表示主动结束读取，不是正常收到 `done`。随后重新运行一条完整请求：

```bash
python -m zero2vllm.serving.client
```

仍应得到 [12,3,8] 和 `done`。查询健康状态可观察当前流与额度：

```bash
curl http://127.0.0.1:8000/health
```

PowerShell 对应 `curl.exe`。所有请求收尾后，应见 `open_streams`=0、`used_tokens`=0、`running`=0。断连检测存在延迟，刚关客户端时查询可能短暂看见旧请求，不能仅凭这个瞬间就认定泄漏。

health 只显示当前汇总状态，不公开每个用户的完整生成历史。自动实验会在同一个测试进程里读取服务记录，保存取消原因、实际生成数量和资源归还证据。

## 7. 输入错误与正常零输出

零输出请求：

```bash
python -m zero2vllm.serving.client --max-new-tokens 0
```

应只有 `accepted` 和 `done`，`output_tokens`=0，`forward_calls`=0。它仍是一份合法响应。

非法 ID：

```bash
python -m zero2vllm.serving.client --prompt 99 --max-new-tokens 3
```

应打印 HTTP 422 和参数错误，客户端非零退出。服务不应该因此崩溃；再发一次正常请求仍然可用。

最大上下文是 32。默认 P=6，N=20 时最大实际输入长度为 6+20−1=25，可以运行；N=30 时变为 35，超出范围，应拒绝。不要把“最多生成几个输出”直接等同于“缓存里会写入同样数量的新位置”。

## 8. 切换到 2.6 的前缀后端

先在服务终端按 Ctrl+C，等待关闭完成，再启动：

```bash
python -m zero2vllm.serving.server --backend prefix --step-delay 0.05
```

需要 CPU PyTorch。连续运行两次默认客户端。两次输出应相同；具体命中证据用下方专用案例读取块索引与引用计数：

```bash
python -m scripts.experiments.lesson0207 --case prefix --output-dir .local/runs/part-02-engine/07-http-and-streaming-output/prefix
```

预期 http-prefix-backend 通过。报告中 `acquisitions` 的 `hit_blocks` 长度依次为 0、2，块大小为 2，即第二次复用 4 个输入位置。最后 `active_block_references`=0，但 `idle_cached_blocks` 可以大于 0，这沿用 2.6 的留存规则。

HTTP 输出仍然只包含请求与 token 信息，没有把物理块编号塞进对外生成协议。

## 9. 一次运行完整案例集

默认 14 个案例：

```bash
python -m scripts.experiments.lesson0207
```

同时验证可选前缀后端，共 15 个：

```bash
python -m scripts.experiments.lesson0207 --include-prefix
```

自动实验自行启动并关闭本机随机端口上的服务，不占用 8000，也不需要你提前启动手工服务。默认报告在 .local/runs/part-02-engine/07-http-and-streaming-output/summary.json，日常复跑不会覆盖公开基线。

| 案例 | 验证层 | 主要检查 |
| --- | --- | --- |
| http-normal | 真实 HTTP | 三项结果、连续序号、唯一 `done`、资源归还 |
| http-zero | 真实 HTTP | 零输出也能正常收尾 |
| http-eos | 真实 HTTP | EOS 输出与自然终止 |
| http-input-errors | 真实 HTTP | 400 / 422 / 415，拒绝发生在生成前 |
| http-two-clients | 真实 HTTP | 两条流的 ID 与输出独立 |
| http-disconnect | 真实 HTTP | 一条提前断开，同伴正常完成 |
| bounded-backpressure | 受控 ASGI `send` | Q=2、一项在途、生产暂停、同伴继续 |
| send-timeout | 受控 ASGI `send` | 发送不返回，期限后收尾 |
| error-after-start | 受控 ASGI | HTTP 200 后输出一项，再返回 `error` |
| body-and-admission-limits | 受控 ASGI | 413 / 408 / 503，状态和额度正确 |
| request-deadline | 服务层 | 请求超时，终态不被满队列堵住 |
| slow-consumer | 服务层 | 持续满队列触发取消 |
| service-shutdown | 服务层 | 驱动任务结束，请求与额度清空 |
| sse-byte-fragments | 字节协议 | 任意分块、中文 UTF-8、截断拒绝 |
| http-prefix-backend | 可选真实 HTTP + CPU 张量 | 冷暖匹配、输出一致、共享引用归零 |

报告每个案例都带 `level` 和 `assertions`，顶层 `passed` 只有全部断言通过才为 true。程序失败时退出码非零。

## 10. 单独复现背压和错误

```bash
python -m scripts.experiments.lesson0207 --case controlled --output-dir .local/runs/part-02-engine/07-http-and-streaming-output/controlled
python -m scripts.experiments.lesson0207 --case http --output-dir .local/runs/part-02-engine/07-http-and-streaming-output/http
python -m scripts.experiments.lesson0207 --case protocol --output-dir .local/runs/part-02-engine/07-http-and-streaming-output/protocol
```

case 可取 all、http、controlled、protocol、prefix。受控案例集合有 7 项，真实 HTTP 基础集合有 6 项，协议集合 1 项。include-prefix 会额外加入一次前缀后端案例。

背压案例里重点找这两项：

```text
slow_calls_before_peer = 3
slow_calls_after_peer  = 3
```

A 的第一次 token 发送已经停住，另有两条事件排队。B 运行并完成期间，A 的 `forward_calls` 不再增加；A 的 `peak_queue` 为 2。随后模拟 A 断连，结束后没有开放流或占用额度。

只让一个真实客户端读取很慢，并不一定能马上让服务器发送变慢；小量输出可能先被操作系统接收。这里明确阻塞 ASGI `send`，验证的是程序在该边界上的行为，不是网络吞吐基准。

## 11. 参数怎么影响行为

| 服务端参数 | 单位 / 默认 | 改动后的观察 |
| --- | --- | --- |
| `--port` | 8000 | 改动后同步修改客户端 URL |
| `--backend` | tiny | prefix 复用 2.6 的共享池，需 CPU PyTorch |
| `--queue-size` | 事件数 / 2 | 减小后更早暂停生产；不会减少 prompt KV 的长度 |
| `--max-requests` | 开放流数 / 8 | 达到入口上限返回 503，含生成结束但传输尚未收尾的流 |
| `--max-running` | 运行请求数 / 2 | 还需满足 KV 额度，不能绕过实际容量 |
| `--step-delay` | 秒 / 0 | 人为延迟每轮，方便演示，不用于性能比较 |
| `--request-timeout` | 秒 / 30 | 包含排队的生成截止时间 |
| `--slow-timeout` | 秒 / 5 | 输出队列持续满且未被消费的期限 |
| `--send-timeout` | 秒 / 5 | 一次发送调用的等待上限 |

请求体上限 8192 字节、读取期限 5 秒、心跳间隔 1 秒由 `StreamingApp` 默认参数提供，当前 CLI 没有对应开关。

观察截止时间的一个例子，先重启服务：

```bash
python -m zero2vllm.serving.server --step-delay 0.2 --request-timeout 0.1
```

再运行默认客户端，通常能看到部分输出，之后是 `reason`=`deadline_exceeded` 的 `error`。准确的输出条数依赖调度时序，不固定写死为一条；这里应验收“不能再把它称为正常 `done`，且资源归还”。

## 12. 报告字段怎么读

- `source`：代码、测试、输入与示例的摘要；工作区还未提交时会如实标记 dirty。
- `full_recompute_reference`：独立完整重算的普通输入与分叉输入结果。
- `events` / `streams`：客户端实际拼出的事件；随机 `request_id` 不作为固定答案。
- `record.status`：引擎的 `finished`、`cancelled` 或 `failed`。
- `record.transport_end`：传输为什么收尾，例如 `response_complete`、`client_disconnected`、`send_timeout`。
- `record.output_ids`：引擎实际提交的输出，可能多于提前退出客户端读到的数量。
- `record.send_completed_tokens`：发送调用已返回的 token 事件数，不是客户端已显示的确认。
- `record.peak_queue`：该请求队列实际达到的峰值；`accepted` 也占队列位置。
- `engine_events`：已提交输出、取消与 `finalized` 的过程记录。
- `assertions`：该案例真正用于通过判定的条件；不要只看截图里的几个数值。

本次公开数据为 CPU 正确性与所有权验证，没有测量 TTFT、GPU token/s 或服务吞吐。

## 13. 测试与常见排查

```bash
python -m unittest tests.serving.test_streaming tests.serving.test_http -v
```

完整环境下 29 项通过；没有 Uvicorn 时 3 项真实 HTTP 检查跳过，没有 PyTorch 时 1 项前缀共享检查跳过。跳过不代表相应能力已经验证。

| 现象 | 先检查 |
| --- | --- |
| Connection refused | 服务终端是否仍运行、端口是否相同、是否完成启动 |
| Address already in use | 旧服务是否仍占端口；关闭旧服务或换端口 |
| 没有 uvicorn 模块 | 安装与运行是否用了同一个虚拟环境 |
| 一口气显示所有结果 | 客户端是否读完整体才打印；curl 是否用了 -N；是否经过代理 |
| 收到 200 后又有 `error` | 看流内 `reason`；200 不是整段生成成功的证明 |
| 提前退出后服务器多生成了几项 | 网络和同步前向存在边界延迟；检查最终取消与额度归零 |
| `used_tokens`=0 但进程内存没下降 | 额度、预分配池、Python 分配器不是同一个指标 |
| 前缀后端结束后仍有缓存块 | 检查活跃引用是否为 0；闲置完整块可留存复用 |
| 队列容量为 2，却生成了 3 项 | 另有一条已取出的发送中事件 |
| 没有 `done`，连接就结束了 | 结果不完整；不要把这次输出当作成功完成 |

结束手工实验后，在服务终端按 Ctrl+C，等待关闭日志。先退出客户端再关闭服务也可以；需要验证“关闭时仍有活跃请求”的场景时，使用自动 shutdown 与取消测试。

## 验收产出

保存一份完整案例报告，解释一次正常流、一次提前断连、一次满队列和一次流内错误。能回答事件顺序、资源归还和同伴隔离三个问题，再完成[十四组练习](exercises.md)。
