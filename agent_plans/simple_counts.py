import json

import redis
from bluesky import plan_stubs as bps


def simple_ct(*args, **kwargs):
    ...


pe1c = ...
Grid_X = ...
Grid_Y = ...
Grid_Z = ...
Det_1_X = ...
Det_1_Y = ...
Det_1_Z = ...
ring_current = ...
BStop1 = ...
get_metadata_for_sample_number = ...
bt = ...


def agent_sample_count(motor, position: float, exposure: float, *, sample_number: int, md=None):
    rkvs = redis.Redis(host="info.pdf.nsls2.bnl.gov", port=6379, db=0)  # redis key value store
    p_my_config = rkvs.get("PDF:xpdacq:user_config")
    user_config = json.loads(p_my_config)
    yield from bps.mv(motor, position)
    _md = dict(
        Grid_X=Grid_X.read(),
        Grid_Y=Grid_Y.read(),
        Grid_Z=Grid_Z.read(),
        Det_1_X=Det_1_X.read(),
        Det_1_Y=Det_1_Y.read(),
        Det_1_Z=Det_1_Z.read(),
        ring_current=ring_current.read(),
        BStop1=BStop1.read(),
        user_config=user_config,
    )
    _md.update(get_metadata_for_sample_number(bt, sample_number))
    _md.update(md or {})
    yield from simple_ct([pe1c], exposure, md=_md)
