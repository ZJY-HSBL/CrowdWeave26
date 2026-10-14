"""CrowdHuman metadata audit regression tests."""
import json
import pytest
from crowdweave.audit import audit_metadata


def test_audit_counts_overlap_and_ignored(tmp_path):
    persons={"a":[{"fbox":[0,0,10,10]},{"fbox":[2,2,10,10]}],
             "b":[{"fbox":[50,50,3,3]}]}
    ignored={"a":[{"fbox":[2,2,2,2]}],"b":[]}
    (tmp_path/"associated_train.json").write_text(json.dumps(persons))
    (tmp_path/"ignored_train.json").write_text(json.dumps(ignored))
    result=audit_metadata(tmp_path)
    assert result["images"]==2 and result["valid_persons"]==3
    assert result["overlap_images"]==1 and result["ignored_images"]==1
    assert result["ignored_touching_persons"]==2


def test_audit_rejects_mismatched_ids(tmp_path):
    (tmp_path/"associated_train.json").write_text('{"a":[]}')
    (tmp_path/"ignored_train.json").write_text('{}')
    with pytest.raises(ValueError):
        audit_metadata(tmp_path)
