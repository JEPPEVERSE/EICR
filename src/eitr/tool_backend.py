"""Explicit OpenAI-compatible native tool transport with redacted accounting."""
import json
import os
import urllib.error
import urllib.request
from time import perf_counter
from eicr.contracts import ContractError, digest, require
from eicr.provider import ChatBackend, ProviderTransportError, _open_no_redirect, _strict_json


class ToolBackend(ChatBackend):
    def chat_tools(self, messages, tools):
        key = os.environ.get(self.key_env)
        if not key: raise ProviderTransportError('Missing credential environment variable: '+self.key_env)
        payload = {'model': self.model, 'messages': messages, 'tools': tools, 'tool_choice': 'auto',
                   'max_tokens': self.max_output_tokens, **self.generation_options}
        trace = {'model': self.model, 'endpoint_hash': digest(self.base_url), 'prompt_hash': digest(messages),
                 'tools_hash': digest(tools), 'generation_options': self.generation_options,
                 'max_output_tokens': self.max_output_tokens, 'status': 'started'}
        self.calls.append(trace); start = perf_counter()
        req = urllib.request.Request(self.base_url+'/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
        try:
            with _open_no_redirect(req, timeout=self.timeout) as response:
                body = response.read(self.max_response_bytes+1)
            require(len(body) <= self.max_response_bytes, 'Tool response exceeds byte cap')
            result = _strict_json(body)
            require(isinstance(result, dict), 'Invalid tool completion')
            trace['response_model']=result.get('model') if isinstance(result.get('model'),str) else None
            trace['system_fingerprint']=result.get('system_fingerprint') if isinstance(result.get('system_fingerprint'),str) else None
            usage = result.get('usage', {})
            trace['usage'] = {k: v for k, v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens')
                              and type(v) is int and v >= 0} if isinstance(usage,dict) else {}
            choices = result.get('choices')
            require(isinstance(choices,list) and choices and isinstance(choices[0],dict), 'Invalid tool choices')
            choice = choices[0]
            require(choice.get('finish_reason') in ('stop','tool_calls'), 'Incomplete tool completion')
            msg = choice.get('message')
            require(isinstance(msg,dict) and msg.get('role') == 'assistant', 'Invalid tool message')
            require(msg.get('content') is None or isinstance(msg['content'],str), 'Invalid tool text')
            calls = msg.get('tool_calls') or []
            require(isinstance(calls,list), 'Invalid tool calls')
            ids = set()
            for call in calls:
                require(isinstance(call,dict) and isinstance(call.get('id'),str) and call['id'] and call['id'] not in ids
                        and call.get('type') == 'function' and isinstance(call.get('function'),dict)
                        and isinstance(call['function'].get('name'),str) and isinstance(call['function'].get('arguments'),str), 'Invalid tool call')
                ids.add(call['id'])
            trace['status']='ok'
            return {k: msg[k] for k in ('role','content','tool_calls') if k in msg}
        except ContractError:
            trace['status']='error'; raise
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            trace['status']='error'
            raise ProviderTransportError('Tool transport failed ('+type(error).__name__+')') from None
        except (ValueError, TypeError, KeyError, IndexError) as error:
            trace['status']='error'
            raise ContractError('Tool completion failed ('+type(error).__name__+')') from None
        finally:
            trace['seconds']=perf_counter()-start
