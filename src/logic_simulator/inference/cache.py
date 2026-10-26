import copy
import json
from pathlib import Path
from ..run_store import append_json


class Cache:
    def __init__(self,path):
        self.path=Path(path)
        self.records={}
        self.kinds={}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                record=json.loads(line)
                self.records[record['key']]=record['result']
                provenance=record['result']['provenance']
                self.kinds.setdefault(record['stage'],set()).add(provenance.get('source_kind',provenance['provider']))

    def store(self,key,result,stage='test',request=None):
        self.records[key]=copy.deepcopy(result)
        provenance=result['provenance']
        self.kinds.setdefault(stage,set()).add(provenance.get('source_kind',provenance['provider']))
        append_json(self.path,{'schema_version':1,'key':key,'stage':stage,'request':request,'result':result})

    def lookup(self,key,request_id):
        if key not in self.records:
            return None
        result=copy.deepcopy(self.records[key])
        result['provenance']['original_request_id']=result['request_id']
        result['provenance']['provider']='replay'
        result['request_id']=request_id
        return result
