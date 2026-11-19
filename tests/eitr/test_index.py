from copy import deepcopy
import numpy as np
import pytest
from eicr.contracts import ContractError
from eitr.native import canonicalize, validate_snapshot, import_pack
from eitr.index import build_index, save_index, load_index


def test_profile_shared_cues_bridge_disjoint_facts(snapshot, tmp_path):
    index = build_index(snapshot)
    pos = {t['text']: i for i, t in enumerate(snapshot['tags'])}
    cp = index.cue_pos
    H = index.profiles.toarray()
    assert H[pos['shopping'], cp['herbs']] == pytest.approx(1/3)
    assert H[pos['cooking'], cp['herbs']] == pytest.approx(1/2)
    assert np.all(H[:, cp['I']] == 0)
    assert np.allclose(H.sum(axis=1), 1)
    t = {t['text']: t['tag_id'] for t in snapshot['tags']}
    assert t['cooking'] in [x['target'] for x in index.neighbors[t['shopping']]]
    assert index.neighbors[t['travel']] == []
    save_index(index, tmp_path/'i.json')
    loaded = load_index(tmp_path/'i.json')
    assert loaded.index_hash == index.index_hash
    assert (loaded.vectors != index.vectors).nnz == 0


def test_source_group_duplication_does_not_gain_mass(raw_memory):
    before = build_index(canonicalize(raw_memory))
    f = deepcopy(raw_memory['evidence'][0]); f['evidence_id'] = 'f1-copy'
    raw_memory['evidence'].append(f)
    raw_memory['edges'] += [{**e, 'evidence_id': 'f1-copy'} for e in list(raw_memory['edges']) if e['evidence_id'] == 'f1']
    after = build_index(canonicalize(raw_memory))
    assert np.allclose(before.profiles.toarray(), after.profiles.toarray())


def test_single_tag_and_no_cues_are_direct_readable(raw_memory):
    for f in raw_memory['evidence']: f['tags'] = ['one']
    index = build_index(canonicalize(raw_memory))
    assert index.report['zero_vectors'] == 1
    raw_memory['edges'] = []
    index = build_index(canonicalize(raw_memory))
    assert index.vectors.nnz == 0


@pytest.mark.parametrize('mutation', ['source', 'edge', 'date', 'owner', 'duplicate'])
def test_reject_broken_native_contract(raw_memory, mutation):
    if mutation == 'source': raw_memory['evidence'][0]['origins'] = ['missing']
    if mutation == 'edge': raw_memory['edges'][0]['cue_id'] = 'missing'
    if mutation == 'date': raw_memory['sources'][0]['datetime_provenance'] = 'rewrite_guessed'
    if mutation == 'owner': raw_memory['sources'][0]['owner_id'] = 'other'
    if mutation == 'duplicate': raw_memory['evidence'].append(raw_memory['evidence'][0])
    with pytest.raises(ContractError): canonicalize(raw_memory)


def test_hash_and_native_triples_validated(snapshot, tmp_path):
    broken = deepcopy(snapshot); broken['facts'][0]['origins'] = ['missing']
    with pytest.raises(ContractError): validate_snapshot(broken)
    index = build_index(snapshot); path=tmp_path/'i.json'; save_index(index,path)
    text = path.read_text().replace('shopping','changed')
    path.write_text(text)
    with pytest.raises(ContractError): load_index(path)


def test_fact_vector_alignment_and_source_balance(snapshot):
    ids = [f['fact_id'] for f in snapshot['facts']]
    bundle = {'snapshot_hash': snapshot['content_hash'], 'encoder':'fixture-v1', 'fact_ids':ids,
              'vectors':[[1,0], [1,0], [0,1], [0,1]]}
    index = build_index(snapshot, fact_vectors=bundle)
    assert index.representation == 'R_fact'
    bundle['fact_ids'] = ids[::-1]
    with pytest.raises(ContractError): build_index(snapshot, fact_vectors=bundle)


def test_import_requires_preinjection_attestation():
    pack={'raw_text':{'D1':{'D1:1':'A: hello'}},'rewrites':[{'session_1':{'sentence':[{'id':'f1','text':'hello','origin':['D1:1'],'tag':'greeting'}]}}],
          'keywords':[{'sentence':[{'sentence_id':'f1','keyword':['hello']}]}]}
    manifest={'owner_id':'x','snapshot_id':'s','allowed_source_ids':['D1:1'],'extraction_prefix_verified':True}
    with pytest.raises(ContractError): import_pack(pack,manifest)
    manifest.update(keyword_cache_is_pre_injection=True,session_datetimes={'D1':'2024-01-01'})
    data=import_pack(pack,manifest)
    assert data['sources'][0]['session_datetime']=='2024-01-01'
    manifest['session_datetimes'] = {'session_1': '2024-01-01'}
    with pytest.raises(ContractError): import_pack(pack, manifest)


def test_alias_requires_in_snapshot_provenance(raw_memory):
    raw_memory['cues'][0]['aliases'] = [{'text': 'smoked pepper', 'source_ids': ['missing']}]
    with pytest.raises(ContractError): canonicalize(raw_memory)
    raw_memory['cues'][0]['aliases'][0]['source_ids'] = ['D1:1']
    validate_snapshot(canonicalize(raw_memory))
