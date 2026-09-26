"""OpenAI-compatible mock of the vLLM server, for local dry runs without a GPU.

Every chat completion answers with one `python` tool call that plays a random
valid action, so the full Duck harness loop (prompting, tool sandbox, action
execution, scoring) can be exercised end to end.
"""
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CODE = (
    "import random\n"
    "acts = [a for a in valid_actions if a != 'RESET'] or valid_actions\n"
    "a = random.choice(acts)\n"
    "if a == 'MOUSE':\n"
    "    r = action([{'action': 'MOUSE', 'row': random.randrange(64), 'col': random.randrange(64)}])\n"
    "else:\n"
    "    r = action([a])\n"
    "print(a, r.get('level'), r.get('state'))\n"
)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self.server.requests += 1
        if req.get("tools"):
            msg = {"role": "assistant", "content": "", "reasoning_content": "mock",
                   "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": CODE})}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "4"}
            finish = "stop"
        self._send({"id": "cmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion", "model": req.get("model", "mock"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.requests = 0
    srv.serve_forever()
