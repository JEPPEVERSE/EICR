"""Explicit USD rates keyed by service, requested model and endpoint identity.

Replay preserves deployment cost. Actual experimental spend must use actual
provider calls, never cached deployment traces. Missing prices/usage stay unknown.
"""
from collections import defaultdict
import math
from eicr.contracts import digest, require

KINDS=('chat_tokens','embedding_tokens','rerank_tokens')


def validate_prices(prices):
    if prices is None:return None
    require(isinstance(prices,dict),'Invalid price table')
    if set(prices)=={'input_per_million','output_per_million'}:
        require(all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in prices.values()),'Invalid legacy prices')
        return 'legacy'
    require(set(prices)=={'schema','currency','as_of','source','rates'} and
            prices['schema']=='eitr-prices-1' and prices['currency']=='USD' and
            all(isinstance(prices[k],str) and prices[k].strip() for k in ('as_of','source')) and
            isinstance(prices['rates'],list),'Invalid service price table')
    seen=set()
    for rate in prices['rates']:
        require(isinstance(rate,dict) and set(rate)=={'billing_kind','model','endpoint_hash',
                'input_per_million','output_per_million','cached_input_per_million'},'Pin complete service rate')
        require(rate['billing_kind'] in KINDS and isinstance(rate['model'],str) and rate['model'] and
                isinstance(rate['endpoint_hash'],str) and len(rate['endpoint_hash'])==64 and
                all(c in '0123456789abcdef' for c in rate['endpoint_hash']),'Invalid service identity')
        key=tuple(rate[k] for k in ('billing_kind','model','endpoint_hash'))
        require(key not in seen,'Duplicate service rates');seen.add(key)
        for k in ('input_per_million','output_per_million','cached_input_per_million'):
            v=rate[k]
            require((k=='cached_input_per_million' and v is None) or
                    (type(v) in (int,float) and math.isfinite(v) and v>=0),'Invalid service rate')
    return 'services'


def price_calls(calls,prices=None):
    mode=validate_prices(prices);rows=[]
    rates={tuple(r[k] for k in ('billing_kind','model','endpoint_hash')):r for r in prices['rates']} if mode=='services' else {}
    for call in calls:
        kind=call.get('billing_kind','chat_tokens');key=(kind,call.get('model'),call.get('endpoint_hash'))
        rate=(prices if mode=='legacy' and kind=='chat_tokens' else rates.get(key))
        u=call.get('usage') or {};i=u.get('prompt_tokens',u.get('input_tokens'));o=u.get('completion_tokens',u.get('output_tokens'))
        error=None;usd=None
        if rate is None:error='missing_price'
        elif not all(type(v) is int and v>=0 for v in (i,o)):error='missing_usage'
        else:
            discount=rate.get('cached_input_per_million')
            cached=u.get('cached_input_tokens')
            if discount is not None and discount!=rate['input_per_million']:
                if type(cached) is not int or not 0<=cached<=i:error='missing_or_invalid_provider_cache_usage'
            if error is None:
                cached=cached if discount is not None and type(cached) is int else 0
                if not 0<=cached<=i:error='invalid_provider_cache_usage'
                else:usd=((i-cached)*rate['input_per_million']+cached*(discount if discount is not None else rate['input_per_million'])+
                          o*rate['output_per_million'])/1e6
        rows.append({'billing_kind':kind,'model':key[1],'endpoint_hash':key[2],'usd':usd,'error':error,
                     'failed_call':call.get('status')!='ok'})
    groups=defaultdict(list)
    for row in rows:groups[(row['billing_kind'],row['model'],row['endpoint_hash'])].append(row)
    services=[]
    for key,items in sorted(groups.items(),key=lambda kv:str(kv[0])):
        complete=all(r['usd'] is not None for r in items)
        services.append({'billing_kind':key[0],'model':key[1],'endpoint_hash':key[2],'calls':len(items),
                         'usd':sum(r['usd'] for r in items) if complete else None,
                         'known_usd':sum(r['usd'] for r in items if r['usd'] is not None)})
    complete=all(r['usd'] is not None for r in rows)
    return {'currency':'USD','prices_hash':digest(prices) if prices is not None else None,
            'usd':sum(r['usd'] for r in rows) if complete else None,
            'known_usd':sum(r['usd'] for r in rows if r['usd'] is not None),
            'complete':complete,'unpriced_calls':sum(r['usd'] is None for r in rows),
            'services':services,'calls':rows,
            'note':'known_usd is a subtotal, not a complete bill. Rates are user-supplied, not fetched or inferred.'}
